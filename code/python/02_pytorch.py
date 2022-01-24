import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt
import random

# ==============================================================================
# Define Dataset & DataLoader
# ==============================================================================
class SyntheticData(Dataset):
    def __init__(self, w, b, num_examples):
        n = Normal(0, 1)
        X = n.sample((num_examples, len(w)))
        y = X @ w + b

        n_eps = Normal(0, 0.01)
        eps = n_eps.sample((num_examples,))
        y += eps

        self.X = X
        self.y = y.reshape(-1, 1)
    
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

true_w = torch.tensor([2, -3.4])
true_b = torch.tensor(4.2)
num_examples = 1000

train_data = SyntheticData(true_w, true_b, num_examples)
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)

# ==============================================================================
# Define Neural Network
# ==============================================================================
class LinReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 1)
        )
    
    def forward(self, X):
        return self.net(X)

# ==============================================================================
# Define Loss & Optimizer
# ==============================================================================
net = LinReg()
loss = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.03)

# ==============================================================================
# Training
# ==============================================================================
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in train_loader:
        y_hat = net(X)
        l = loss(y_hat, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    l = loss(net(train_data.X), train_data.y)
    print(f'epoch {epoch + 1}: loss {l.item():f}')