from torch.utils.data import Dataset
import json
import soundfile as sf
import torch
import numpy as np
from pathlib import Path
from asteroid.data.librimix_dataset import librispeech_license


class TACDataset(Dataset):
    """Multi-channel Librispeech-derived dataset used in Transform Average Concatenate.

    Args:
        json_file (str): Path to json file resulting from the data prep script which contains parsed examples.
        segment (float, optional): Length of the segments used for training, in seconds.
            If None, use full utterances (e.g. for test).
        sample_rate (int, optional): The sampling rate of the wav files.
        max_mics (int, optional): Maximum number of microphones for an array in the dataset.
        train (bool, optional): If True randomly permutes the microphones on each example.
    """

    dataset_name = "TACDataset"

    def __init__(self, json_file, segment=None, sample_rate=16000, max_mics=6, train=True):
        self.segment = segment
        self.sample_rate = sample_rate
        self.max_mics = max_mics
        self.train = train

        with open(json_file, "r") as f:
            examples = json.load(f)

        if self.segment:
            target_len = int(segment * sample_rate)
            self.examples = []
            for ex in examples:
                if ex["1"]["length"] < target_len:
                    continue
                self.examples.append(ex)
            print(
                "Discarded {} out of {} because too short".format(
                    len(examples) - len(self.examples), len(examples)
                )
            )
        else:
            self.examples = examples
        if not train:
            # sort examples based on number
            self.examples = sorted(
                self.examples, key=lambda x: str(Path(x["2"]["spk1"]).parent).strip("sample")
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        """Returns mixtures, sources and the number of mics in the recording, padded to `max_mics`."""
        c_ex = self.examples[item]
        # randomly select ref mic
        mics = [x for x in c_ex.keys()]
        if self.train:
            np.random.shuffle(mics)  # randomly permute during training to change ref mics

        mixtures = []
        sources = []
        for i in range(len(mics)):
            c_mic = c_ex[mics[i]]

            if self.segment:
                offset = 0
                if c_mic["length"] > int(self.segment * self.sample_rate):
                    offset = np.random.randint(
                        0, c_mic["length"] - int(self.segment * self.sample_rate)
                    )

                # we load mixture
                mixture, fs = sf.read(
                    c_mic["mixture"],
                    start=offset,
                    stop=offset + int(self.segment * self.sample_rate),
                    dtype="float32",
                )
                spk1, fs = sf.read(
                    c_mic["spk1"],
                    start=offset,
                    stop=offset + int(self.segment * self.sample_rate),
                    dtype="float32",
                )
                spk2, fs = sf.read(
                    c_mic["spk2"],
                    start=offset,
                    stop=offset + int(self.segment * self.sample_rate),
                    dtype="float32",
                )
            else:
                mixture, fs = sf.read(c_mic["mixture"], dtype="float32")  # load all
                spk1, fs = sf.read(c_mic["spk1"], dtype="float32")
                spk2, fs = sf.read(c_mic["spk2"], dtype="float32")

            mixture = torch.from_numpy(mixture).unsqueeze(0)
            spk1 = torch.from_numpy(spk1).unsqueeze(0)
            spk2 = torch.from_numpy(spk2).unsqueeze(0)

            assert fs == self.sample_rate
            mixtures.append(mixture)
            sources.append(torch.cat((spk1, spk2), 0))

        mixtures = torch.cat(mixtures, 0)
        sources = torch.stack(sources)
        # we pad till max_mic
        valid_mics = mixtures.shape[0]
        if mixtures.shape[0] < self.max_mics:
            dummy = torch.zeros((self.max_mics - mixtures.shape[0], mixtures.shape[-1]))
            mixtures = torch.cat((mixtures, dummy), 0)
            sources = torch.cat((sources, dummy.unsqueeze(1).repeat(1, sources.shape[1], 1)), 0)
        return mixtures, sources, valid_mics

    def get_infos(self):
        """Get dataset infos (for publishing models).

        Returns:
            dict, dataset infos with keys `dataset`, `task` and `licences`.
        """
        infos = dict()
        infos["dataset"] = self.dataset_name
        infos["task"] = "separate_noisy"
        infos["licenses"] = [librispeech_license, tac_license]
        return infos


tac_license = dict(
    title="End-to-end Microphone Permutation and Number Invariant Multi-channel Speech Separation",
    title_link="https://arxiv.org/abs/1910.14104",
    author="Yi Luo, Zhuo Chen, Nima Mesgarani, Takuya Yoshioka",
    license="CC BY 4.0",
    license_link="https://creativecommons.org/licenses/by/4.0/",
    non_commercial=False,
)
