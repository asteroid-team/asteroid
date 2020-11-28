import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import soundfile as sf


class FUSSDataset(Dataset):
    """Dataset class for FUSS [1] tasks.

    Args:
        file_list_path (str): Path to the txt (csv) file created at stage 2
            of the recipe.
        return_bg (bool): Whether to return the background along the mixture
            and sources (useful for SIR, SAR computation). Default: False.

    References
        [1] Scott Wisdom et al. "What's All the FUSS About Free Universal
        Sound Separation Data?", 2020, in preparation.
    """

    dataset_name = "FUSS"

    def __init__(self, file_list_path, return_bg=False):
        super().__init__()
        # Arguments
        self.return_bg = return_bg
        # Constants
        self.max_n_fg = 3
        self.n_src = self.max_n_fg  # Same variable as in WHAM
        self.sample_rate = 16000
        self.num_samples = self.sample_rate * 10

        # Load the file list as a dataframe
        # FUSS has a maximum of 3 foregrounds, make column names
        self.fg_names = [f"fg{i}" for i in range(self.max_n_fg)]
        names = ["mix", "bg"] + self.fg_names
        # Lines with less labels will have nan, replace with empty string
        self.mix_df = pd.read_csv(file_list_path, sep="\t", names=names)
        # Number of foregrounds (fg) vary from 0 to 3
        # This can easily be used to remove mixtures with less than x fg
        # remove_less_than = 1
        # self.mix_df.dropna(threshold=remove_less_than, inplace=True)
        # self.mix_df.reset_index(inplace=True)
        self.mix_df.fillna(value="", inplace=True)

    def __len__(self):
        return len(self.mix_df)

    def __getitem__(self, idx):
        # Each line has absolute to miture, background and foregrounds
        line = self.mix_df.iloc[idx]
        mix = sf.read(line["mix"], dtype="float32")[0]
        sources = []
        for fg_path in [line[fg_n] for fg_n in self.fg_names]:
            if fg_path:
                sources.append(sf.read(fg_path, dtype="float32")[0])
            else:
                sources.append(np.zeros_like(mix))
        sources = torch.from_numpy(np.vstack(sources))

        if self.return_bg:
            bg = sf.read(line["bg"], dtype="float32")[0]
            return torch.from_numpy(mix), sources, torch.from_numpy(bg)
        return torch.from_numpy(mix), sources

    def get_infos(self):
        """Get dataset infos (for publishing models).

        Returns:
            dict, dataset infos with keys `dataset`, `task` and `licences`.
        """
        infos = dict()
        infos["dataset"] = self.dataset_name
        infos["task"] = "sep_noisy"
        infos["licenses"] = [fuss_license]
        return infos


fuss_license = dict(
    title="Free Universal Sound Separation Dataset",
    title_link="https://zenodo.org/record/3743844#.X0Jtehl8Jkg",
    author="Scott Wisdom; Hakan Erdogan; Dan Ellis and John R. Hershey",
    author_link="https://scholar.google.com/citations?user=kJM6N7IAAAAJ&hl=en",
    license="Creative Commons Attribution 4.0 International",
    license_link="https://creativecommons.org/licenses/by/4.0/legalcode",
    non_commercial=False,
)
