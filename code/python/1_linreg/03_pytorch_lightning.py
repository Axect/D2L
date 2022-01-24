import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.distributions import Normal

import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningModule, seed_everything

import numpy as np
import matplotlib.pyplot as plt

AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 100 if AVAIL_GPUS else 10

# ==============================================================================
# Set seed
# ==============================================================================
seed_everything(8407)

# ==============================================================================
# Define Dataset
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

# ==============================================================================
# Define LightningModule
# ==============================================================================
class LinReg(LightningModule):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 1)
        )
    
    def forward(self, X):
        return self.net(X)
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self(X)
        loss = F.mse_loss(y_pred, y)
        self.log('train_loss', loss, prog_bar=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self(X)
        loss = F.mse_loss(y_pred, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self(X)
        loss = F.mse_loss(y_pred, y)
        self.log('test_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.SGD(self.net.parameters(), lr=0.03)
        return optimizer

    def prepare_data(self):
        true_w = torch.tensor([2, -3.4])
        true_b = torch.tensor(4.2)
        num_examples = 1200
        self.ds = SyntheticData(true_w, true_b, num_examples)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.ds_train, self.ds_val = random_split(self.ds, [1000, 200])
        if stage == "test" or stage is None:
            _, self.ds_test = random_split(self.ds, [1000, 200])

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=10, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=10)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=10)

# ==============================================================================
# Train & Validation
# ==============================================================================
model = LinReg()

trainer = Trainer(
    max_epochs = 3,
    gpus = AVAIL_GPUS,
)

trainer.fit(model)

trainer.test()