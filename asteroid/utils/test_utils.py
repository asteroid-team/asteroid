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


class DummyWaveformDataset(data.Dataset):
    def __init__(self, total=12, n_src=3, len_wave=16000):
        self.inp_len_wave = len_wave
        self.out_len_wave = len_wave
        self.total = total
        self.inp_n_sig = 1
        self.out_n_sig = n_src

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        mixed = torch.randn(self.inp_n_sig, self.inp_len_wave)
        srcs = torch.randn(self.out_n_sig, self.out_len_wave)
        return mixed, srcs


def torch_version_tuple():
    version, *suffix = torch.__version__.split("+")
    return tuple(map(int, version.split("."))) + tuple(suffix)
