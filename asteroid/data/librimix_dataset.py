import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data.dataset import Dataset


class LibriMix(Dataset):
    """" Dataset class for Librimix source separation tasks.

    Args:
        metadata_path (str): The path to the metatdata file in
        dataset/metadata
        desired_length (int) : The desired sources and mixtures length in s
        sample_rate (int) : The sample rate of the sources and mixtures
        n_src (int) : The number of sources in the mixture
    """

    def __init__(self, metadata_path, desired_length=None, sample_rate=16000,
                 n_src=2):
        self.metadata_path = metadata_path
        self.desired_length = desired_length
        self.sample_rate = sample_rate
        # Open csv file
        self.df = pd.read_csv(metadata_path)
        if self.desired_length is not None:
            self.desired_frames = self.desired_length * self.sample_rate
            # Ignore the file shorter than the desired_length
            self.df = self.df[self.df['length'] >= self.desired_frames]
        self.n_src = n_src

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mixture_path = row['mixture_path']
        sources_list = []
        # Read sources
        for i in range(self.n_src):
            source_path = row[f'source_{i+1}_path']
            s, _ = sf.read(source_path, dtype='float32')
            if self.desired_length is not None:
                s = s[:self.desired_frames]
            sources_list.append(s)

        mixture, _ = sf.read(mixture_path, dtype='float32')
        if self.desired_length is not None:
            mixture = mixture[:self.desired_frames]

        mixture = torch.from_numpy(mixture).type(
            'torch.FloatTensor').unsqueeze(0)

        sources = np.vstack(sources_list)
        sources = torch.from_numpy(sources).type('torch.FloatTensor')
        return mixture, sources
