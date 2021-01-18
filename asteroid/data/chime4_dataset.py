import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
import random as random
import os


class CHiME4(Dataset):
    """Dataset class for CHiME4 source separation tasks. Only supports 'real'
    data

    Args:
        csv_dir (str): The path to the metadata file.
        sample_rate (int) : The sample rate of the sources and mixtures.
        segment (int) : The desired sources and mixtures length in s.

    References
        Emmanuel Vincent, Shinji Watanabe, Aditya Arie Nugraha, Jon Barker, and Ricard Marxer
        An analysis of environment, microphone and data simulation mismatches in robust speech recognition
        Computer Speech and Language, 2017.
    """

    dataset_name = "CHiME4"

    def __init__(self, csv_dir, sample_rate=16000, segment=3, return_id=False):
        self.csv_dir = csv_dir
        # Get the csv corresponding to origin
        self.segment = segment
        self.sample_rate = sample_rate
        self.return_id = return_id
        self.csv_path = [f for f in os.listdir(csv_dir) if 'annotations' not in f][0]
        # Open csv file and concatenate them
        self.df = pd.read_csv(os.path.join(csv_dir,self.csv_path))
        # Get rid of the utterances too short
        if self.segment is not None:
            max_len = len(self.df)
            self.seg_len = int(self.segment * self.sample_rate)
            # Ignore the file shorter than the desired_length
            self.df = self.df[self.df["duration"] >= self.seg_len]
            print(
                f"Drop {max_len - len(self.df)} utterances from {max_len} "
                f"(shorter than {segment} seconds)"
            )
        else:
            self.seg_len = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the row in dataframe
        row = self.df.iloc[idx]
        # Get mixture path
        self.mixture_path = row["mixture_path"]
        # If there is a seg start point is set randomly
        if self.seg_len is not None:
            start = random.randint(0, row["length"] - self.seg_len)
            stop = start + self.seg_len
        else:
            start = 0
            stop = None

        # Read the mixture
        mixture, _ = sf.read(self.mixture_path, dtype="float32",
                             start=start, stop=stop)
        # Convert to torch tensor
        mixture = torch.from_numpy(mixture)
        fake_source = torch.vstack([mixture])
        if self.return_id:
            id1 = row.ID
            return mixture, fake_source, [id1]
        return mixture, fake_source

