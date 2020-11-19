import torch
from torch.utils import data
import json
import os
import soundfile as sf

from asteroid.utils import get_wav_random_start_stop


class DNSDataset(data.Dataset):
    """Deep Noise Suppression (DNS) Challenge's dataset.

    Args
        json_dir (str): path to the JSON directory (from the recipe).

    References
        - "The INTERSPEECH 2020 Deep Noise Suppression Challenge: Datasets,
        Subjective Testing Framework, and Challenge Results", Reddy et al. 2020.
    """

    dataset_name = "DNS"

    def __init__(self, json_dir, sample_rate=16000, segment=None, load_noise=True):
        super(DNSDataset, self).__init__()
        self.sample_rate = sample_rate
        self.segment = segment
        self.load_noise = load_noise
        self.json_dir = json_dir
        with open(os.path.join(json_dir, "file_infos.json"), "r") as f:
            self.mix_infos = json.load(f)

        self.wav_ids = list(self.mix_infos.keys())

    def __len__(self):
        return len(self.wav_ids)

    def __getitem__(self, idx):
        """Gets a mixture/sources pair.
        Returns:
            (mixture, clean, noise) if self.load_noise is true, else (mixture, clean)
        """
        utt_info = self.mix_infos[self.wav_ids[idx]]
        # Load mixture
        x_np, sr = sf.read(utt_info["mix"], dtype="float32")
        assert sr == self.sample_rate
        start, stop = get_wav_random_start_stop(
            len(x_np), int(self.segment * self.sample_rate) if self.segment is not None else None
        )
        x = torch.from_numpy(x_np[start:stop])
        # Load clean
        speech = torch.from_numpy(
            sf.read(utt_info["clean"], dtype="float32", start=start, stop=stop)[0]
        )
        if self.segment is not None:
            x = torch.nn.functional.pad(
                x, (0, max(0, int(self.segment * self.sample_rate) - len(x)))
            )
            speech = torch.nn.functional.pad(
                speech, (0, max(0, int(self.segment * self.sample_rate) - len(speech)))
            )
        if self.load_noise:
            # Load noise
            noise = torch.from_numpy(
                sf.read(utt_info["noise"], dtype="float32", start=start, stop=stop)[0]
            )
            if self.segment is not None:
                noise = torch.nn.functional.pad(
                    noise, (0, max(0, int(self.segment * self.sample_rate) - len(noise)))
                )
            return x, speech, noise
        else:
            return x, speech

    def get_infos(self):
        """Get dataset infos (for publishing models).

        Returns:
            dict, dataset infos with keys `dataset`, `task` and `licences`.
        """
        return {
            "dataset": self.dataset_name,
            "sample_rate": self.sample_rate,
            "segment": self.segment,
            "task": "enhancement",
            "licenses": [dns_license],
            "n_src": 2 if self.load_noise else 1,
        }


dns_license = dict(
    title="Deep Noise Suppression (DNS) Challenge",
    title_link="https://github.com/microsoft/DNS-Challenge",
    author="Microsoft",
    author_link="https://www.microsoft.com/fr-fr/",
    license="CC BY-NC 4.0",
    license_link="https://creativecommons.org/licenses/by-nc/4.0/",
    non_commercial=False,
)
