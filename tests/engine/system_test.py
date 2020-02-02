# Test the maximum of functions, very small model will be fast to train for one
# batch

# Need to test training, and loading right?

import torch
from torch import nn, optim
from torch.utils import data
from pytorch_lightning import Trainer

from asteroid import System


class DummyDataset(data.Dataset):
    def __init__(self):
        self.inp_dim = 10
        self.out_dim = 10

    def __len__(self):
        return 20

    def __getitem__(self, idx):
        return torch.randn(1, self.inp_dim), torch.randn(1, self.out_dim)


def test_system():
    model = nn.Sequential(nn.Linear(10, 10), nn.ReLU())
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    dataset = DummyDataset()
    loader = data.DataLoader(dataset, batch_size=2)
    system = System(model, optimizer, loss_func=nn.MSELoss(),
                    train_loader=loader, val_loader=loader,
                    scheduler=scheduler)
    trainer = Trainer(max_epochs=1, fast_dev_run=True)
    trainer.fit(system)
