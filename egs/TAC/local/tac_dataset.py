from torch.utils.data import Dataset
import json
import soundfile as sf
import torch
import numpy as np
from pathlib import Path


class TACDataset(Dataset):
    def __init__(self, json_file, segment=None, samplerate=16000, max_mics=6, train=True):

        self.segment = segment
        self.samplerate = samplerate
        self.max_mics = max_mics
        self.train = train

        with open(json_file, "r") as f:
            examples = json.load(f)

        if self.segment:
            target_len = int(segment * samplerate)
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
                if c_mic["length"] > int(self.segment * self.samplerate):
                    offset = np.random.randint(
                        0, c_mic["length"] - int(self.segment * self.samplerate)
                    )

                # we load mixture
                mixture, fs = sf.read(
                    c_mic["mixture"],
                    start=offset,
                    stop=offset + int(self.segment * self.samplerate),
                    dtype="float32",
                )

                spk1, fs = sf.read(
                    c_mic["spk1"],
                    start=offset,
                    stop=offset + int(self.segment * self.samplerate),
                    dtype="float32",
                )

                spk2, fs = sf.read(
                    c_mic["spk2"],
                    start=offset,
                    stop=offset + int(self.segment * self.samplerate),
                    dtype="float32",
                )
            else:
                mixture, fs = sf.read(c_mic["mixture"], dtype="float32")  # load all
                spk1, fs = sf.read(c_mic["spk1"], dtype="float32")
                spk2, fs = sf.read(c_mic["spk2"], dtype="float32")

            mixture = torch.from_numpy(mixture).unsqueeze(0)
            spk1 = torch.from_numpy(spk1).unsqueeze(0)
            spk2 = torch.from_numpy(spk2).unsqueeze(0)

            assert fs == self.samplerate
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

        return mixtures, sources, valid_mics, c_ex
