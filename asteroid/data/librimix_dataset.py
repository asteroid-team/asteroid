import os
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data.dataset import Dataset
import pyloudnorm as pyln


class LibriMix(Dataset):
    """" Dataset class for Librimix source separation tasks.

    Args:
        metadata_file_path (str): The path to the metatdata file in
        dataset/metadata
        mixtures_directory_path (str): The path to the flac file in
        dataset/freq/mode/directory
        sample_length (int) : The desired sources and mixtures length in s
        n_src (int) : The number of sources in the mixture
    """

    def __init__(self, sources_directory_path, mixtures_directory_path,
                 sample_length=4):

        self.sources_directory_path = sources_directory_path
        self.mixtures_directory_path = mixtures_directory_path
        # Define the frame length
        self.sample_length = sample_length
        self.sample_frame = self.sample_length * 16000
        # list the wav file in the directory
        self.file_list = os.listdir(self.mixtures_directory_path)
        self.path_list = []
        # Ignore the file shorter than the sample_length
        for file in self.file_list:
            file_path = os.path.join(self.mixtures_directory_path, file)
            if self.sample_length is not None:
                if len(sf.SoundFile(file_path)) >= self.sample_frame:
                    self.path_list.append(file_path)
            else:
                self.path_list.append(file_path)

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):

        mixture_path = self.path_list[idx]
        source_ID_list = os.path.split(mixture_path)[1].strip('.wav').split(
            '_')
        sources_list = []
        # Read sources
        for source_ID in source_ID_list:
            s, _ = sf.read(os.path.join(self.sources_directory_path,
                                        source_ID + '.wav'))
            sources_list.append(s)

        mixture, _ = sf.read(mixture_path)

        mixture = torch.from_numpy(mixture).type(
            'torch.FloatTensor').unsqueeze(0)

        sources = np.vstack(sources_list)
        sources = torch.from_numpy(sources).type('torch.FloatTensor')
        return mixture, sources
