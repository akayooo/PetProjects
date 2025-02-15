import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_eval_loop(model, optimizer, train_loader, val_loader, criterion, lr=-1e9, epoch=10, device='cpu'):

    model.to(device)
    for epoch in range(epoch):
        model.train()
        total_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for param in model.parameters():
                    param = -lr * param.grad

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader) 
        print(f"Epoch [{epoch+1}/{epoch}], Train Loss: {avg_train_loss:.4f}")

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        
        val_acc = 100 * correct / total
        print(f"Validation Accuracy: {val_acc:.2f}%\n")


if __name__ == "__main__":
    # Создание простых данных (пример)
    x_train = torch.randn(100, 10)
    y_train = torch.randint(0, 2, (100,))
    x_val = torch.randn(20, 10)
    y_val = torch.randint(0, 2, (20,))
    
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 2)
    )

    lr = 0.01

    criterion = nn.CrossEntropyLoss()  # Функция потерь
    optimizer = optim.SGD(model.parameters(), lr=lr)  # Оптимизатор SGD
    
    train_eval_loop(model, optimizer, train_loader, val_loader, criterion, lr=lr,  epoch=5, device=device)

