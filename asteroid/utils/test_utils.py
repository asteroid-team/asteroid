import torch
from torch.utils import data


class DummyDataset(data.Dataset):
    def __init__(self):
        self.inp_dim = 10
        self.out_dim = 10

    def __len__(self):
        return 20

    def __getitem__(self, idx):
        return torch.randn(1, self.inp_dim), torch.randn(1, self.out_dim)
