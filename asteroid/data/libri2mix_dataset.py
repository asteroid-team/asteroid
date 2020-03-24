# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 18:52:47 2020

@author: user
"""
import os
import pandas as pd
import numpy as np
import soundfile as sf
import torch
from torch.utils.data.dataset import Dataset
from scipy.signal import resample_poly
import random


class Libri2Mix(Dataset):
    """" Dataset class for Libri2mix source separation tasks.

    Args:
        metadata_file_path (str): The path to the metatdata file in
        mixtures/metadata

        sample_rate (int) : The desired sample rate in Hz

        mode (str) : not sure yet

        sample_length (int) : The desired sources and mixtures length in s
    """

    def __init__(self, metadata_file_path, sample_rate=16000, mode='max',
                 sample_length=1):

        # Get the  metadata file:
        self.metadata_file = pd.read_csv(os.path.join(metadata_file_path))
        self.sample_rate = sample_rate
        self.mode = mode
        self.sample_length = sample_length
        self.sample_frame = self.sample_length * 16000

        if sample_length is not None:
            self.metadata_file = self.metadata_file[
                (self.metadata_file['S1_Length'] >= self.sample_frame) |
                (self.metadata_file['S2_Length'] >= self.sample_frame)]

    def __len__(self):
        return len(self.metadata_file)

    def __getitem__(self, idx):

        mixture_path = self.metadata_file.iloc[idx]['Mixtures_path']
        s1_path = self.metadata_file.iloc[idx]['S1_path']
        s2_path = self.metadata_file.iloc[idx]['S2_path']

        if self.sample_length is not None:

            start1 = random.randint(0,
                                    self.metadata_file.iloc[idx]['S1_Length']
                                    - self.sample_frame)
            start2 = random.randint(0,
                                    self.metadata_file.iloc[idx]['S2_Length']
                                    - self.sample_frame)
            s1, rate = sf.read(os.path.join(s1_path), dtype="float32",
                               start=start1, stop=start1 + self.sample_frame)
            s2, rate = sf.read(os.path.join(s2_path), dtype="float32",
                               start=start2, stop=start2 + self.sample_frame)
            mixture = s1 + s2

            if self.sample_rate != 16000:
                mixture = resample_poly(mixture, self.sample_rate, rate)
                s1 = resample_poly(s1, self.sample_rate, rate)
                s2 = resample_poly(s2, self.sample_rate, rate)

        else:

            mixture, rate = sf.read(mixture_path, dtype="float32")
            s1, rate = sf.read(os.path.join(s1_path), dtype="float32")
            s2, rate = sf.read(os.path.join(s2_path), dtype="float32")

            if self.sample_rate != 16000:
                mixture = resample_poly(mixture, self.sample_rate, rate)
                s1 = resample_poly(s1, self.sample_rate, rate)
                s2 = resample_poly(s2, self.sample_rate, rate)

        mixture = torch.from_numpy(mixture).type(
            'torch.FloatTensor').unsqueeze(0)

        sources_array = [s1, s2]
        sources = np.vstack(sources_array)
        sources = torch.from_numpy(sources).type('torch.FloatTensor')
        return mixture, sources
