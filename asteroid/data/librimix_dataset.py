import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch import hub
from torch.utils.data import Dataset, DataLoader
import random as random
import os
import shutil
import zipfile

from .wham_dataset import wham_noise_license

MINI_URL = "https://zenodo.org/record/3871592/files/MiniLibriMix.zip?download=1"


class LibriMix(Dataset):
    """Dataset class for LibriMix source separation tasks.

    Args:
        csv_dir (str): The path to the metadata file.
        task (str): One of ``'enh_single'``, ``'enh_both'``, ``'sep_clean'`` or
            ``'sep_noisy'`` :

            * ``'enh_single'`` for single speaker speech enhancement.
            * ``'enh_both'`` for multi speaker speech enhancement.
            * ``'sep_clean'`` for two-speaker clean source separation.
            * ``'sep_noisy'`` for two-speaker noisy source separation.

        sample_rate (int) : The sample rate of the sources and mixtures.
        n_src (int) : The number of sources in the mixture.
        segment (int, optional) : The desired sources and mixtures length in s.

    References
        [1] "LibriMix: An Open-Source Dataset for Generalizable Speech Separation",
        Cosentino et al. 2020.
    """

    dataset_name = "LibriMix"

    def __init__(
        self, csv_dir, task="sep_clean", sample_rate=16000, n_src=2, segment=3, return_id=False
    ):
        self.csv_dir = csv_dir
        self.task = task
        self.return_id = return_id
        # Get the csv corresponding to the task
        if task == "enh_single":
            md_file = [f for f in os.listdir(csv_dir) if "single" in f][0]
            self.csv_path = os.path.join(self.csv_dir, md_file)
        elif task == "enh_both":
            md_file = [f for f in os.listdir(csv_dir) if "both" in f][0]
            self.csv_path = os.path.join(self.csv_dir, md_file)
            md_clean_file = [f for f in os.listdir(csv_dir) if "clean" in f][0]
            self.df_clean = pd.read_csv(os.path.join(csv_dir, md_clean_file))
        elif task == "sep_clean":
            md_file = [f for f in os.listdir(csv_dir) if "clean" in f][0]
            self.csv_path = os.path.join(self.csv_dir, md_file)
        elif task == "sep_noisy":
            md_file = [f for f in os.listdir(csv_dir) if "both" in f][0]
            self.csv_path = os.path.join(self.csv_dir, md_file)
        self.segment = segment
        self.sample_rate = sample_rate
        # Open csv file
        self.df = pd.read_csv(self.csv_path)
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
        self.n_src = n_src

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
            start = random.randint(0, row["length"] - self.seg_len)
            stop = start + self.seg_len
        else:
            start = 0
            stop = None
        # If task is enh_both then the source is the clean mixture
        if "enh_both" in self.task:
            mix_clean_path = self.df_clean.iloc[idx]["mixture_path"]
            s, _ = sf.read(mix_clean_path, dtype="float32", start=start, stop=stop)
            sources_list.append(s)

        else:
            # Read sources
            for i in range(self.n_src):
                source_path = row[f"source_{i + 1}_path"]
                s, _ = sf.read(source_path, dtype="float32", start=start, stop=stop)
                sources_list.append(s)
        # Read the mixture
        mixture, _ = sf.read(mixture_path, dtype="float32", start=start, stop=stop)
        # Convert to torch tensor
        mixture = torch.from_numpy(mixture)
        # Stack sources
        sources = np.vstack(sources_list)
        # Convert sources to tensor
        sources = torch.from_numpy(sources)
        if not self.return_id:
            return mixture, sources
        # 5400-34479-0005_4973-24515-0007.wav
        id1, id2 = mixture_path.split("/")[-1].split(".")[0].split("_")
        return mixture, sources, [id1, id2]

    @classmethod
    def loaders_from_mini(cls, batch_size=4, **kwargs):
        """Downloads MiniLibriMix and returns train and validation DataLoader.

        Args:
            batch_size (int): Batch size of the Dataloader. Only DataLoader param.
                To have more control on Dataloader, call `mini_from_download` and
                instantiate the DatalLoader.
            **kwargs: keyword arguments to pass the `LibriMix`, see `__init__`.
                The kwargs will be fed to both the training set and validation
                set.

        Returns:
            train_loader, val_loader: training and validation DataLoader out of
            `LibriMix` Dataset.

        Examples
            >>> from asteroid.data import LibriMix
            >>> train_loader, val_loader = LibriMix.loaders_from_mini(
            >>>     task='sep_clean', batch_size=4
            >>> )
        """
        train_set, val_set = cls.mini_from_download(**kwargs)
        train_loader = DataLoader(train_set, batch_size=batch_size, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, drop_last=True)
        return train_loader, val_loader

    @classmethod
    def mini_from_download(cls, **kwargs):
        """Downloads MiniLibriMix and returns train and validation Dataset.
        If you want to instantiate the Dataset by yourself, call
        `mini_download` that returns the path to the path to the metadata files.

        Args:
            **kwargs: keyword arguments to pass the `LibriMix`, see `__init__`.
                The kwargs will be fed to both the training set and validation
                set

        Returns:
            train_set, val_set: training and validation instances of
            `LibriMix` (data.Dataset).

        Examples
            >>> from asteroid.data import LibriMix
            >>> train_set, val_set = LibriMix.mini_from_download(task='sep_clean')
        """
        # kwargs checks
        assert "csv_dir" not in kwargs, "Cannot specify csv_dir when downloading."
        assert kwargs.get("task", "sep_clean") in [
            "sep_clean",
            "sep_noisy",
        ], "Only clean and noisy separation are supported in MiniLibriMix."
        assert (
            kwargs.get("sample_rate", 8000) == 8000
        ), "Only 8kHz sample rate is supported in MiniLibriMix."
        # Download LibriMix in current directory
        meta_path = cls.mini_download()
        # Create dataset instances
        train_set = cls(os.path.join(meta_path, "train"), sample_rate=8000, **kwargs)
        val_set = cls(os.path.join(meta_path, "val"), sample_rate=8000, **kwargs)
        return train_set, val_set

    @staticmethod
    def mini_download():
        """Downloads MiniLibriMix from Zenodo in current directory

        Returns:
            The path to the metadata directory.
        """
        mini_dir = "./MiniLibriMix/"
        os.makedirs(mini_dir, exist_ok=True)
        # Download zip (or cached)
        zip_path = mini_dir + "MiniLibriMix.zip"
        if not os.path.isfile(zip_path):
            hub.download_url_to_file(MINI_URL, zip_path)
        # Unzip zip
        cond = all([os.path.isdir("MiniLibriMix/" + f) for f in ["train", "val", "metadata"]])
        if not cond:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall("./")  # Will unzip in MiniLibriMix
        # Reorder metadata
        src = "MiniLibriMix/metadata/"
        for mode in ["train", "val"]:
            dst = f"MiniLibriMix/metadata/{mode}/"
            os.makedirs(dst, exist_ok=True)
            [
                shutil.copyfile(src + f, dst + f)
                for f in os.listdir(src)
                if mode in f and os.path.isfile(src + f)
            ]
        return "./MiniLibriMix/metadata"

    def get_infos(self):
        """Get dataset infos (for publishing models).

        Returns:
            dict, dataset infos with keys `dataset`, `task` and `licences`.
        """
        infos = dict()
        infos["dataset"] = self._dataset_name()
        infos["task"] = self.task
        if self.task == "sep_clean":
            data_license = [librispeech_license]
        else:
            data_license = [librispeech_license, wham_noise_license]
        infos["licenses"] = data_license
        return infos

    def _dataset_name(self):
        """ Differentiate between 2 and 3 sources."""
        return f"Libri{self.n_src}Mix"


librispeech_license = dict(
    title="LibriSpeech ASR corpus",
    title_link="http://www.openslr.org/12",
    author="Vassil Panayotov",
    author_link="https://github.com/vdp",
    license="CC BY 4.0",
    license_link="https://creativecommons.org/licenses/by/4.0/",
    non_commercial=False,
)
