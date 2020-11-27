import torch
from asteroid.data.librimix_dataset import LibriMix
import os
from torch.utils.data import DataLoader
import torch.nn.functional as F


def collate_fn(batch):

    inputs = []
    targets = []

    if batch is not None:
        max_len = 0
        for elm in batch:
            mixture, sources = elm
            if len(mixture) > max_len:
                max_len = len(mixture)
        for elm in batch:
            mixture, sources = elm
            mixture = F.pad(mixture, pad=(0, max_len - len(mixture)),
                           mode='constant', value=0)
            sources = F.pad(sources, pad=(0, max_len - sources.shape[-1]),
                                mode='constant', value=0)
            inputs.append(mixture)
            targets.append(sources)
    inputs = torch.vstack(inputs)
    targets = torch.vstack(targets)
    return inputs, targets


test_dir='../../egs/librimix/ConvTasNet/data/wav8k/max/test'

test_set = LibriMix(
    csv_dir=test_dir,
    task='sep_clean',
    sample_rate=8000,
    n_src=2,
    segment=None)

test_loarder = DataLoader(
        test_set,
        shuffle=False,
        batch_size=4,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_fn)



