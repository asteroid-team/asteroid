import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data.dataset import Dataset
import random as random


class LibriMix(Dataset):
    """" Dataset class for Librimix source separation tasks.

    Args:
        metadata_path (str): The path to the metatdata file
        sample_rate (int) : The sample rate of the sources and mixtures
        n_src (int) : The number of sources in the mixture
        segment (int) : The desired sources and mixtures length in s
    """

    def __init__(self, metadata_path, sample_rate=16000,
                 n_src=2, segment='full'):
        self.metadata_path = metadata_path
        self.segment = segment
        self.sample_rate = sample_rate
        # Open csv file
        self.df = pd.read_csv(metadata_path)
        if self.segment != 'full':
            self.seg_len = self.segment * self.sample_rate
            # Ignore the file shorter than the desired_length
            self.df = self.df[self.df['length'] >= self.seg_len]
        else:
            self.seg_len = None
        self.n_src = n_src

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mixture_path = row['mixture_path']
        sources_list = []
        if self.seg_len is not None:
            start = random.randint(0, row['length'] - self.seg_len)
            stop = start + self.seg_len
        else:
            start = 0
            stop = row['length']

        # Read sources
        for i in range(self.n_src):
            source_path = row[f'source_{i+1}_path']
            s, _ = sf.read(source_path, dtype='float32', start=start,
                           stop=stop)
            sources_list.append(s)

        mixture, _ = sf.read(mixture_path, dtype='float32', start=start,
                             stop=stop)

        mixture = torch.from_numpy(mixture)

        sources = np.vstack(sources_list)
        sources = torch.from_numpy(sources)
        return mixture, sources
