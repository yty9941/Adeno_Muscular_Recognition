import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import config
class EarlyStopping:
    def __init__(self):
        self.patience = config.Early_Stopping_Patience
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# model = nn.Linear(10, 1)
# criterion = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01)
#
# early_stopping = EarlyStopping()
#
# for epoch in range(100):
#     train_loss = 0.0
#     val_loss = 0.0
#
#     for batch, (inputs, targets) in enumerate(train_loader):
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()
#
#     for batch, (inputs, targets) in enumerate(val_loader):
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
#         val_loss += loss.item()
#
#     train_loss /= len(train_loader)
#     val_loss /= len(val_loader)
#
#     print(f"Epoch {epoch+1}: Train Loss = {train_loss}, Val Loss = {val_loss}")
#
#     early_stopping(val_loss)
#     if early_stopping.early_stop:
#         print("Early stopping")
#         break
