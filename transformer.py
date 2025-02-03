import pandas as pd
import numpy as np
import re
import spacy
from tqdm import tqdm
from datasets import load_dataset
from evaluate import load
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_scheduler
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import seaborn as sns

tqdm.pandas()


df = pd.read_csv('transformer.csv')
data = df[['Review Text', 'Recommended IND']]
data.columns = ['text', 'labels']


nlp = spacy.load("en_core_web_sm")
def preprocess_text(text):

    text = str(text)
    text = text.lower()
    text = re.sub(r'@\w+|#[\w-]+|http\S+|\n', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc]

    return ' '.join(lemmas)


data_cleared = data.copy()
data_cleared['text'] = data['text'].progress_apply(preprocess_text)
data_cleared.to_csv('cleared_transformer.csv', index=False)


model_name = "BERT-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


full_dataset = load_dataset('csv', data_files='cleared_transformer.csv')
full_dataset = full_dataset.filter(lambda example: example['text'] is not None and example['labels'] is not None)
tokenizer.tokenize(full_dataset['train'][1]['text'])


dataset = full_dataset['train'].train_test_split(test_size=0.2)
dataset = dataset.map(
    lambda e: tokenizer(  
        e['text'],  
        truncation=True,  
        max_length=80,  
        padding='max_length' 
    ),
    batched=True 
)


dataset = dataset.remove_columns('text')
dataset.set_format(type='torch', device='cuda')
train_dataloader = DataLoader(dataset['train'], shuffle=True, batch_size=16)
test_dataloader = DataLoader(dataset['test'], shuffle=False, batch_size=8)


optimizer = AdamW(model.parameters(), lr=1e-4)
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "cosine",  
    optimizer=optimizer,  
    num_warmup_steps=int(0.1 * num_epochs * len(train_dataloader)),  
    num_training_steps=num_training_steps  
)


best_f1 = 0. 
device = 'cuda' if torch.cuda.is_available() else 'cpu' 

running_loss = []  
for epoch in range(num_epochs):
    print(40*'-', f'\nЭпоха {epoch+1}')

    model.train()  
    pbar = tqdm(train_dataloader, total=len(train_dataloader), desc='Обучение')

    for i, batch in enumerate(pbar):
        batch = {k: v.to(device) for k, v in batch.items()}  

        outputs = model(**batch)  
        loss = outputs.loss  

        loss.backward()  
        optimizer.step()  
        lr_scheduler.step() 
        optimizer.zero_grad()  

        running_loss.append(loss.item()) 
        pbar.set_postfix({'running loss': np.mean(running_loss[-25:])}) 
    
    print('\nВалидация')
    model.eval()  

    f1 = load('f1', trust_remote_code=True)
    acc = load('accuracy', trust_remote_code=True)
    precision = load('precision', trust_remote_code=True)
    recall = load('recall', trust_remote_code=True)

    val_los = [] 
    pbar = tqdm(test_dataloader, total=len(test_dataloader), desc='Валидация')

    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():  
            outputs = model(**batch)  
        
        logits = outputs.logits.detach().cpu()  
        predictions = torch.argmax(logits, dim=-1)  
        references = batch['labels'].detach().cpu()  

        f1.add_batch(predictions=predictions, references=references)
        acc.add_batch(predictions=predictions, references=references)
        precision.add_batch(predictions=predictions, references=references)
        recall.add_batch(predictions=predictions, references=references)

        if hasattr(outputs, 'loss'):  
            val_los.append(outputs.loss.item()) 
        pbar.set_postfix({'val_los': np.mean(val_los[-25:])})  

    print('Итоговые метрики (взвешенные):')
    test_acc = acc.compute()['accuracy']  
    test_precision = precision.compute(average='weighted')['precision']  
    test_recall = recall.compute(average='weighted')['recall']  
    f1_weighted = f1.compute(average='weighted')['f1']  
    print(f'[{epoch+1}] Accuracy: {test_acc:.4f}')
    print(f'[{epoch+1}] Precision: {test_precision:.4f}')
    print(f'[{epoch+1}] Recall: {test_recall:.4f}')
    print(f'[{epoch+1}] F1: {f1_weighted:.4f}')

    if f1_weighted > best_f1:  
        best_f1 = f1_weighted  
        model.save_pretrained(f"best_model_nov_2021_f1_max={best_f1}_len=100")  
        print(f"Новое лучшее F1: {best_f1}. Модель сохранена.")

