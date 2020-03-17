# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 18:52:47 2020

@author: user
"""

import os

import numpy as np
import soundfile as sf
import torch
from torch.utils.data.dataset import Dataset


class Libri2Mix(Dataset):
    """" Dataset class for Libri2mix source separation tasks.

    Args:
        directory (str): The path to the directory 16K or 8K containing the
        folder 'sources' and 'mixtures'

        mode (str): The mode to be taken for data loading
    """
    def __init__(self, directory, mode="train"):

        # setting directories for data
        data_root = os.path.join(directory, 'mixtures')
        sep_path = os.path.join(directory, 'sources')

        self.mode = mode
        if self.mode == "train":
            self.data_dir = os.path.join(data_root, "train")
            self.sep_path = os.path.join(sep_path, "train")
        elif self.mode == "test":
            self.data_dir = os.path.join(data_root, "test")
            self.sep_path = os.path.join(sep_path, "test")
        elif self.mode == "cv":
            self.data_dir = os.path.join(data_root, "cv")
            self.sep_path = os.path.join(sep_path, "cv")

        self.mixture_list = os.listdir(self.data_dir)

    def __len__(self):
        return len(os.listdir(self.data_dir))

    def __getitem__(self, idx):
        mixture = self.mixture_list[idx]
        s1_path = os.path.join(self.sep_path, mixture.split("_")[0])
        s2_path = os.path.join(self.sep_path, mixture.split("_")[1][:-5])

        mixture, rate = sf.read(os.path.join(self.data_dir, mixture),
                                dtype="float32")
        s1, rate = sf.read(os.path.join(s1_path), dtype="float32")
        s2, rate = sf.read(os.path.join(s2_path), dtype="float32")
        mixture = torch.from_numpy(mixture).type(
            'torch.FloatTensor').unsqueeze(0)
        sources_array = [s1, s2]
        sources = np.vstack(sources_array)
        sources = torch.from_numpy(sources).type('torch.FloatTensor')
        return mixture, sources
