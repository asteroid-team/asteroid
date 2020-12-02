import torch
from asteroid.data.librimix_dataset import LibriMix
import os
from torch.utils.data import DataLoader
import torch.nn.functional as F


def collate_fn(batch):

    inputs = []
    targets = []
    sizes = []
    if batch is not None:
        max_len = 0
        for elm in batch:
            mixture, sources = elm
            if len(mixture) > max_len:
                max_len = len(mixture)
        for elm in batch:
            mixture, sources = elm
            sizes.append(torch.tensor(len(mixture)))
            mixture = F.pad(mixture, pad=(0, max_len - len(mixture)),
                           mode='constant', value=0)
            sources = F.pad(sources, pad=(0, max_len - sources.shape[-1]),
                                mode='constant', value=0)
            inputs.append(mixture)
            targets.append(sources.unsqueeze(0))
    inputs = torch.vstack(inputs)
    targets = torch.vstack(targets)
    sizes = torch.vstack(sizes)
    return inputs, targets, sizes


