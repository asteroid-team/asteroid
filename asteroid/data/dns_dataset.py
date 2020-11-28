import torch
from torch.utils import data
import json
import os
import soundfile as sf


class DNSDataset(data.Dataset):
    """Deep Noise Suppression (DNS) Challenge's dataset.

    Args
        json_dir (str): path to the JSON directory (from the recipe).

    References
        "The INTERSPEECH 2020 Deep Noise Suppression Challenge: Datasets,
        Subjective Testing Framework, and Challenge Results", Reddy et al. 2020.
    """

    dataset_name = "DNS"

    def __init__(self, json_dir):

        super(DNSDataset, self).__init__()
        self.json_dir = json_dir
        with open(os.path.join(json_dir, "file_infos.json"), "r") as f:
            self.mix_infos = json.load(f)

        self.wav_ids = list(self.mix_infos.keys())

    def __len__(self):
        return len(self.wav_ids)

    def __getitem__(self, idx):
        """Gets a mixture/sources pair.
        Returns:
            mixture, vstack([source_arrays])
        """
        utt_info = self.mix_infos[self.wav_ids[idx]]
        # Load mixture
        x = torch.from_numpy(sf.read(utt_info["mix"], dtype="float32")[0])
        # Load clean
        speech = torch.from_numpy(sf.read(utt_info["clean"], dtype="float32")[0])
        # Load noise
        noise = torch.from_numpy(sf.read(utt_info["noise"], dtype="float32")[0])
        return x, speech, noise

    def get_infos(self):
        """Get dataset infos (for publishing models).

        Returns:
            dict, dataset infos with keys `dataset`, `task` and `licences`.
        """
        infos = dict()
        infos["dataset"] = self.dataset_name
        infos["task"] = "enhancement"
        infos["licenses"] = [dns_license]
        return infos


dns_license = dict(
    title="Deep Noise Suppression (DNS) Challenge",
    title_link="https://github.com/microsoft/DNS-Challenge",
    author="Microsoft",
    author_link="https://www.microsoft.com/fr-fr/",
    license="CC BY-NC 4.0",
    license_link="https://creativecommons.org/licenses/by-nc/4.0/",
    non_commercial=False,
)
