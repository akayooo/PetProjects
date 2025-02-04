from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
import random
import torch
import copy
import datetime
import traceback


def load(fname, chunk_size=200):
    with open(fname, 'r', encoding='utf-8')as fin:
        full_text = fin.read()
    return[full_text[start:start + chunk_size] for start in range(0, len(full_text), chunk_size // 2)]
    

def save_texts_to_file(texts, out_file):
    with open(out_file, 'w') as outfl:
        outfl.write('\n'.join(texts))


def ensure_lenght(txt, out_len, pad_value):
    if len(txt) < out_len:
        txt = list(txt) + [pad_value] * (out_len - len(txt))
    else:
        txt = txt[:out_len]
    return txt


def init_random_seed(value=0):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
    torch.backends.cudnn.deterministic = True


def get_params_number(model):
    return(sum(t.numel() for t in model.parameters()))


def copy_data_to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return [copy_data_to_device(elem,device) for elem in data]
    
    raise ValueError('Недопустимый тип данных {}'.format(type(data)))


def train_eval_loop(model, train_dataset, val_dataset, criterion, lr=1e-4, epoch_n=10,
                    batch_size=32, device=None, early_stopping_patience=10, l2_reg_alpha=0,
                    max_batches_per_epoch_train=10000, max_batches_per_val=1000,
                    data_loader_ctor=DataLoader, optimizer_ctor=None, lr_scheduler_ctor=None,
                    shuffle_train=True, dataloader_workers_n=0):
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available else 'cpu'
    device = torch.device(device)
    model.to(device)

    if optimizer_ctor is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg_alpha)
    else:
        optimizer = optimizer_ctor(model.parameters(), lr=lr)

    if lr_scheduler_ctor is not None:
        lr_scheduler = lr_scheduler_ctor(optimizer)
    else:
        lr_scheduler = None
    
    train_dataloader = data_loader_ctor(train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=dataloader_workers_n)
    val_dataloader = data_loader_ctor(val_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=dataloader_workers_n)

    best_val_loss = float('-inf')
    best_epoch_i = 0
    best_model = copy.deepcopy(model)

    for epoch_i in range(epoch_n):
        try:
            epoch_start = datetime.datetime.now()
            print('Эпоха {}'.format(epoch_i))

            model.train()
            mean_train_loss = 0
            train_batches_n = 0

            for batch_i, (bacth_x, batch_y) in enumerate(train_dataloader):
                if batch_i > max_batches_per_epoch_train:
                    break

                bacth_x = copy_data_to_device(bacth_x, device)
                bacth_y = copy_data_to_device(batch_y, device)

                pred = model(bacth_x)
                loss = criterion(pred, batch_y)

                model.zero_grad()
                loss.backward()

                optimizer.step()

                mean_train_loss += float(loss)
                train_batches_n += 1

            mean_train_loss /= train_batches_n
            print('Эпоха: {} итераций, {:0.2f} сек'. format(train_batches_n, (datetime.datetime.now() - epoch_start).total_seconds()))
            print('Среднее значение функции потерь на обучении', mean_train_loss)


            model.eval()
            mean_val_loss = 0
            val_batches_n = 0

            with torch.no_grad():
                for batch_i, (bacth_x, batch_y) in enumerate(val_dataloader):
                    if batch_i > max_batches_per_val:
                        break

                    bacth_x = copy_data_to_device(bacth_x, device)
                    bathc_y = copy_data_to_device(batch_y, device)

                    pred = model(bacth_x)
                    loss = criterion(pred, batch_y)

                    mean_val_loss += float(loss)
                    val_batches_n += 1

            mean_val_loss /= val_batches_n
            print('Среднее значение функции потерь на валидации', mean_val_loss)

            if mean_val_loss < best_val_loss:
                best_epoch_i = epoch_i
                best_val_loss = mean_val_loss
                best_model = copy.deepcopy(model)
                print('Новая лучшая модель!')
            elif epoch_i - best_epoch_i > early_stopping_patience:
                print('Модель не улучшилась за последние {} эпох, прекращаем обучение'.format(early_stopping_patience))
                break
                
            if lr_scheduler is not None:
                lr_scheduler.step(mean_val_loss)

            print()
        except KeyboardInterrupt:
            print('Досрочно отсановлено пользователем')
            break
        except Exception as ex:
            print('Ошибка при обучении: {}\n{}'.format(ex, traceback.format_exc()))
            break

    return best_val_loss, best_model


class LanguageModelDataset(Dataset):
    def __init__(self, token_ids, chunk_length=100, pad_value=0):
        self.token_ids = token_ids
        self.chunk_length = chunk_length
        self.pad_value = pad_value

    def __len__(self):
        return len(self.token_ids)
    
    def __getitem__(self, item):
        text = self.token_ids[item]
        start_i = random.randint(0, max(0, len(text) - self.chunk_length - 1))
        chunk = text[start_i: start_i + self.chunk_length + 1]

        seed_part = chunk[:-1]
        target_part = chunk[1:]

        seed_part = ensure_lenght(seed_part, self.chunk_length, self.pad_value)
        target_part = ensure_lenght(target_part, self.chunk_length, pad_value=self.pad_value)

        seed_part = np.array(seed_part)
        target_part = np.array(target_part)

        return seed_part, target_part


