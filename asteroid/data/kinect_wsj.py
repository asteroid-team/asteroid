import torch
from torch.utils import data
import json
import os
import numpy as np
import soundfile as sf
from .wsj0_mix import Wsj0mixDataset


def make_dataloaders(train_dir, valid_dir, n_src=2, sample_rate=16000,
                     segment=4.0, batch_size=4, num_workers=None,
                     **kwargs):
    num_workers = num_workers if num_workers else batch_size
    train_set = KinectWsjMixDataset(train_dir, n_src=n_src,
                               sample_rate=sample_rate,
                               segment=segment)
    val_set = KinectWsjMixDataset(valid_dir, n_src=n_src,
                             sample_rate=sample_rate,
                             segment=segment)
    train_loader = data.DataLoader(train_set, shuffle=True,
                                   batch_size=batch_size,
                                   num_workers=num_workers,
                                   drop_last=True)
    val_loader = data.DataLoader(val_set, shuffle=True,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 drop_last=True)
    return train_loader, val_loader


class KinectWsjMixDataset(Wsj0mixDataset):
    def __init__(self, json_dir, n_src=2, sample_rate=16000, segment=4.0):
        super().__init__( json_dir, n_src=n_src, sample_rate=sample_rate, segment=segment)
        noises = []
        for i in range(len(self.mix)):  
            path = self.mix[i][0]
            # Warning: linux specific
            path_splits = path.split('/')
            path_splits[-2] = 'noise'
            noise_path = '/' + os.path.join(*path_splits)
            noises.append([noise_path, self.mix[i][1]])
        self.noises = noises

    def __getitem__(self, idx):
        """ Gets a mixture/sources pair.
        Returns:
            mixture, stack([source_arrays]), noise
            mixture is of dimension [samples, channels]
            sources are of dimension [n_src, samples, channels]
        """
        # Random start
        if self.mix[idx][1] == self.seg_len or self.like_test:
            rand_start = 0
        else:
            rand_start = np.random.randint(0, self.mix[idx][1] - self.seg_len)
        if self.like_test:
            stop = None
        else:
            stop = rand_start + self.seg_len
        # Load mixture
        x, _ = sf.read(self.mix[idx][0], start=rand_start, stop=stop, dtype='float32', always_2d=True)
        noise, _ = sf.read(self.noises[idx][0], start=rand_start, stop=stop, dtype='float32', always_2d=True)
        seg_len = torch.as_tensor([len(x)])
        # Load sources
        source_arrays = []
        for src in self.sources:
            if src[idx] is None:
                # Target is filled with zeros if n_src > default_nsrc
                s = np.zeros_like(x)
            else:
                s, _ = sf.read(src[idx][0], start=rand_start,
                               stop=stop, dtype='float32', always_2d=True)
            source_arrays.append(s)
        sources = torch.from_numpy(np.stack(source_arrays))
        return torch.from_numpy(x), sources, torch.from_numpy(noise)
