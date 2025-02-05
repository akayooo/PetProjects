import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

import numpy as np
import torch
from torch.nn import functional as F

import youtokentome as yttm
import torch.nn as nn

from utils import load, save_texts_to_file, LanguageModelDataset, init_random_seed,\
        get_params_number, train_eval_loop, GreedyGenerator,\
        BeamGenerator, mask_for_attention, positional_encoding, multihead_attention,\
        lm_cross_entropy, lr_sheduler

init_random_seed()


print("CUDA доступна:", torch.cuda.is_available())
print("Кол-во доступных GPU:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Имя GPU:", torch.cuda.get_device_name(0))
    print("Текущий GPU:", torch.cuda.current_device())


all_chunks = load('./data/war_and_peace.txt')
np.random.shuffle(all_chunks)


TRAIN_SPLIT = int(len(all_chunks) * 0.7)
train_texts = all_chunks[:TRAIN_SPLIT]
test_texts = all_chunks[TRAIN_SPLIT:]
print(f'Размеры выборок train/test: \n {len(train_texts)} \n {len(test_texts)}')


BPE_MODEL_FILENAME = './data/war_and_peace_bpe_train.txt'
TRAIN_TEXT_FILENAME = './data/war_and_peace_bpe_train.txt'
save_texts_to_file(train_texts, TRAIN_TEXT_FILENAME)
yttm.BPE.train(data=TRAIN_TEXT_FILENAME, vocab_size=1000, model=BPE_MODEL_FILENAME)
tokenizer = yttm.BPE(BPE_MODEL_FILENAME)


train_token_ids = tokenizer.encode(train_texts, bos=True, eos=True)
test_token_ids = tokenizer.encode(test_texts, bos=True, eos=True)
unknown_words = sum(1 for text in test_token_ids for token_id in text if token_id == 1)
print('Неизвестных слов:', unknown_words)


CHUNK_LENGTH = 80
train_dataset = LanguageModelDataset(token_ids=train_token_ids, chunk_length=CHUNK_LENGTH, pad_value=0)
test_dataset = LanguageModelDataset(token_ids=test_token_ids, chunk_length=CHUNK_LENGTH, pad_value=0)


class LanguageModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size, model, emb_dropout=0.0):
        super().__init__()
        self.embedding_size = embedding_size
        self.embeddings = torch.nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(emb_dropout)
        self.model = model
        self.out = torch.nn.Linear(embedding_size, vocab_size)
    
    def forward(self, seed_token_ids):
        batch_size, max_in_length =  seed_token_ids.shape

        seed_padding_mask = seed_token_ids == 0
        attention_mask = mask_for_attention(max_in_length).to(seed_token_ids.device)

        seed_embs = self.embeddings(seed_token_ids)
        pos_codes = positional_encoding(max_in_length, self.embedding_size).unsqueeze(0).to(seed_embs.device)

        seed_embs = seed_embs + pos_codes
        seed_embs = self.emb_dropout(seed_embs)

        target_features = seed_embs
        target_features = self.model(seed_embs, mask=attention_mask, src_key_padding_mask=seed_padding_mask)

        logits = self.out(target_features)
        return logits
    

class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, model_size, n_heads, dropout=0):
        super().__init__()
        assert model_size % n_heads == 0, 'Размерность модели должна делиться нацело на количество голов'
        self.n_heads = n_heads

        self.queries_proj = nn.Linear(model_size, model_size)
        self.keys_proj = nn.Linear(model_size, model_size)
        self.values_proj = nn.Linear(model_size, model_size)
        
        self.dropout = dropout

        self.last_attention_map = None

    def forward(self, sequence, padding_mask, dependency_mask):
        batch_size, max_len, model_size = sequence.shape

        queries_flat = self.queries_proj(sequence)
        queries = queries_flat.view(batch_size, max_len, self.n_heads, -1)

        keys_flat = self.keys_proj(sequence)
        keys = keys_flat.view(batch_size, max_len, self.n_heads, -1)

        values_flat = self.values_proj(sequence)
        values = values_flat.view(batch_size, max_len, self.n_heads, -1)

        result, att_map = multihead_attention(queries, keys, values,
                                              padding_mask, dependency_mask, self.training, self.dropout)
        
        result_flat = result.view(batch_size, max_len, model_size)

        self.last_attention_map = att_map.detach()

        return result_flat
    

class TransformerEncoderLayer(torch.nn.Module):
    def __init__(self, model_size, n_heads, dim_feedforward, dropout):
        super().__init__()
        self.self_attention = MultiHeadSelfAttention(model_size, n_heads, dropout=dropout)
        self.first_dropout = torch.nn.Dropout(dropout)
        self.first_norm = torch.nn.LayerNorm(model_size)

        self.feedforward = torch.nn.Sequential(
            torch.nn.Linear(model_size, dim_feedforward),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(dim_feedforward, model_size),
            torch.nn.Dropout(dropout)
        )

        self.second_norm = torch.nn.LayerNorm(model_size)

    def forward(self, sequence, padding_mask, dependency_mask):
        att_features = self.self_attention(sequence, padding_mask, dependency_mask)

        sequence = sequence + self.first_dropout(att_features)
        sequence = self.first_norm(sequence)

        sequence = sequence + self.feedforward(sequence)
        sequence = self.second_norm(sequence)

        return sequence
    

class TransformerEncoder(torch.nn.Module):
    def __init__(self, n_layers, **layer_kwargs):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            TransformerEncoderLayer(**layer_kwargs)
            for _ in range(n_layers)
        ])
        self.initialize_weights()

    def forward(self, sequence, mask, src_key_padding_mask):
        for layer in self.layers:
            sequence = layer(sequence, src_key_padding_mask, mask)
        return sequence
    
    def initialize_weights(self):
        for param in self.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)
         

transformer =  LanguageModel(tokenizer.vocab_size(),
                             256,
                             TransformerEncoder(
                                 n_layers=3,
                                 model_size=256,
                                 n_heads=16,
                                 dim_feedforward=512,
                                 dropout=0.1),
                            emb_dropout=0.1
                             )

''' ОБУЧЕНИЕ МОДЕЛИ НА CUDA
print('Количество параметров:', get_params_number(transformer))
(best_val_loss, best_transformer_model) = train_eval_loop(
    transformer,
    train_dataset,
    test_dataset,
    lm_cross_entropy,
    lr=2e-3,
    epoch_n=2000,
    batch_size=256,
    device='cuda',
    early_stopping_patience=50,
    max_batches_per_epoch_train=1000,
    max_batches_per_val=1000,
    lr_scheduler_ctor=lr_sheduler
)
torch.save(best_transformer_model.state_dict(), './models/war_and_peace_best_transformer.pth')
''' 


transformer.load_state_dict(torch.load('./models/war_and_peace_best_transformer.pth'))
my_greedy_generator = GreedyGenerator(transformer, tokenizer)
my_greedy_generator('сказала княжна, оглядывая Андре')


beam_generator = BeamGenerator(transformer, tokenizer)

beam_generator_text = beam_generator('сказала княжна, оглядывая Андре',
                               beamsize=15,
                               return_hypotheses_n=5)

for score, pred_txt in beam_generator_text:
    print('=' * 100)
    print(score)
    print(pred_txt, '\n')