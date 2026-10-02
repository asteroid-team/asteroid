import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset
import random
import os

class VariableLibriMix(Dataset):
    """Dataset class for variable source LibriMix source separation tasks.

    Args:
        csv_dirs (list of str): The paths to the metadata files.
        task (str): One of ``'enh_single'``, ``'enh_both'``, ``'sep_clean'`` or
            ``'sep_noisy'``.
        sample_rate (int) : The sample rate of the sources and mixtures.
        n_src (int) : The maximum number of sources in the mixture (output channels).
        segment (int, optional) : The desired sources and mixtures length in s.
    """

    dataset_name = "VariableLibriMix"

    def __init__(
        self, csv_dirs, task="sep_clean", sample_rate=16000, n_src=5, segment=3, return_id=False
    ):
        self.csv_dirs = csv_dirs if isinstance(csv_dirs, list) else [csv_dirs]
        self.task = task
        self.return_id = return_id
        self.sample_rate = sample_rate
        self.n_src = n_src
        self.segment = segment

        dfs = []
        for csv_dir in self.csv_dirs:
            if task == "enh_single":
                md_file = [f for f in os.listdir(csv_dir) if "single" in f][0]
                csv_path = os.path.join(csv_dir, md_file)
            elif task == "enh_both":
                md_file = [f for f in os.listdir(csv_dir) if "both" in f][0]
                csv_path = os.path.join(csv_dir, md_file)
                # We don't support enh_both mixed with variable sources easily yet
                # because we need the clean file associated with it.
                # Assuming standard LibriMix structure if needed.
            elif task == "sep_clean":
                md_file = [f for f in os.listdir(csv_dir) if "clean" in f][0]
                csv_path = os.path.join(csv_dir, md_file)
            elif task == "sep_noisy":
                md_file = [f for f in os.listdir(csv_dir) if "both" in f][0]
                csv_path = os.path.join(csv_dir, md_file)
            
            df = pd.read_csv(csv_path)
            dfs.append(df)

        self.df = pd.concat(dfs, ignore_index=True)

        # Get rid of the utterances too short
        if self.segment is not None:
            max_len = len(self.df)
            self.seg_len = int(self.segment * self.sample_rate)
            # Ignore the file shorter than the desired_length
            self.df = self.df[self.df["length"] >= self.seg_len]
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
        mixture_path = row["mixture_path"]
        self.mixture_path = mixture_path
        sources_list = []
        # If there is a seg start point is set randomly
        if self.seg_len is not None:
            start = random.randint(0, int(row["length"] - self.seg_len))
            stop = start + self.seg_len
        else:
            start = 0
            stop = None

        # Read the mixture
        mixture, _ = sf.read(mixture_path, dtype="float32", start=start, stop=stop)
        
        # Read sources
        for i in range(self.n_src):
            col_name = f"source_{i + 1}_path"
            if col_name in row and pd.notna(row[col_name]):
                source_path = row[col_name]
                s, _ = sf.read(source_path, dtype="float32", start=start, stop=stop)
            else:
                # Silence
                s = np.zeros_like(mixture)
            sources_list.append(s)

        # Convert to torch tensor
        mixture = torch.from_numpy(mixture)
        # Stack sources
        sources = np.vstack(sources_list)
        # Convert sources to tensor
        sources = torch.from_numpy(sources)
        if not self.return_id:
            return mixture, sources
        
        # Try to parse ID from filename, might fail if formats differ
        try:
            id1, id2 = mixture_path.split("/")[-1].split(".")[0].split("_")
            return mixture, sources, [id1, id2]
        except:
             return mixture, sources, ["unknown", "unknown"]

    def get_infos(self):
        infos = dict()
        infos["dataset"] = self.dataset_name
        infos["task"] = self.task
        return infos
