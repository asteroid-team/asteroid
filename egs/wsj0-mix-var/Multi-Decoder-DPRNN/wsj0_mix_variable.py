"""
Dataset classes for variable number of speakers
Author: Junzhe Zhu
"""
import numpy as np
import torch
import torch.utils.data as data
import soundfile as sf
from time import time
import glob
import os
import random
import json
from tqdm import tqdm


def load_json(filename):
    with open(filename) as f:
        data = json.load(f)
    return data


def pad_audio(audio, len_samples):
    if len(audio) < len_samples:
        audio = np.concatenate([audio, np.zeros(len_samples - len(audio))])
    return audio


class Wsj0mixVariable(data.Dataset):
    """Dataset class for the wsj0-mix with variable number of speakers source separation dataset,

    Args:
        json_dirs: list of folders containing json files, e.g. **/dataset/#speakers/wav8k/min/tr/**
        n_srcs: list specifying number of speakers for each folder
        sample_rate: sample rate
        seglen: length of segment in seconds
        minlen: minimum segment length

    References
        Junzhe Zhu, Raymond Yeh, & Mark Hasegawa-Johnson. (2020). Multi-Decoder DPRNN: High Accuracy Source Counting and Separation.
    """

    def __init__(
        self, json_dirs, n_srcs=[2, 3, 4, 5], sample_rate=8000, seglen=4.0, minlen=2.0
    ):  # segment and cv_maxlen not implemented
        if seglen is None:
            self.seg_len = None
            self.min_len = None
        else:
            self.seg_len = int(seglen * sample_rate)
            self.min_len = int(minlen * sample_rate)
        self.like_test = self.seg_len is None
        self.sr = sample_rate
        self.data = []
        for json_dir, n_src in zip(json_dirs, n_srcs):
            mix_json = os.path.join(json_dir, "mix.json")
            mixfiles, wavlens = list(zip(*load_json(mix_json)))
            sources_json = [
                os.path.join(json_dir, tmp_str + ".json")
                for tmp_str in [f"s{n+1}" for n in range(n_src)]
            ]
            sourcefiles = []
            for source_json in sources_json:
                sourcefiles.append([line[0] for line in load_json(source_json)])
            sourcefiles = list(zip(*sourcefiles))
            self.data += list(zip(mixfiles, sourcefiles, wavlens))

        orig_len = len(self.data)
        drop_utt, drop_len = 0, 0
        if not self.like_test:
            for i in range(len(self.data) - 1, -1, -1):  # Go backward, since we will delete stuff
                if self.data[i][2] < self.min_len:
                    drop_utt += 1
                    drop_len += self.data[i][2]
                    del self.data[i]

        print(
            "Drop {} utts({:.2f} h) from {} (shorter than {} samples)".format(
                drop_utt, drop_len / self.sr / 3600, orig_len, self.min_len
            )
        )

        random.seed(0)
        self.data = random.sample(self.data, len(self.data))
        # Count for resampling
        data_n_src = [len(tmp[1]) for tmp in self.data]
        unique, counts = np.unique(np.array(data_n_src), return_counts=True)
        n_src2counts = dict(zip(unique, counts))
        print("count of mixtures by number of sources:", n_src2counts)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            mixture: [T]
            sources: list of C, each [T]
        """
        mixfile, sourcefiles, length = self.data[idx]
        if self.like_test:
            rand_start = 0
        else:
            rand_start = np.random.randint(0, length - self.min_len + 1)
        if self.like_test:
            stop = None
        else:
            stop = min(rand_start + self.seg_len, length)
        mixture, sr = sf.read(mixfile, start=rand_start, stop=stop, dtype="float32")
        assert sr == self.sr, "need to resample"
        sources = [
            sf.read(sourcefile, start=rand_start, stop=stop, dtype="float32")[0]
            for sourcefile in sourcefiles
        ]
        return mixture, sources


def _collate_fn(batch):
    """
    Args:
        batch: list, len(batch) = batch_size, each entry is a tuple of (mixture, sources)
    Returns:
        mixtures_tensor: B x T, torch.Tensor, padded mixtures
        source_tensor: B x C x T, torch.Tensor, padded in both channel and time dimension
        ilens : B, torch.Tensor, length of each mixture
        num_sources : B, torch.Tensor, number of sources for each mixture
    """
    ilens = [len(mixture) for mixture, _ in batch]
    num_sources = [len(sources) for _, sources in batch]
    mixture_tensor = torch.zeros(len(batch), max(ilens))
    source_tensor = torch.zeros(len(batch), max(num_sources), max(ilens))

    for i, (mixture, sources) in enumerate(batch):  # compute length to pad to
        assert len(mixture) == len(sources[0])
        mixture_tensor[i, : ilens[i]] = torch.Tensor(mixture).float()
        source_tensor[i, : num_sources[i], : ilens[i]] = torch.Tensor(
            np.stack(sources, axis=0)
        ).float()
    ilens = torch.Tensor(np.stack(ilens)).int()
    num_sources = torch.Tensor(np.stack(num_sources)).int()

    return mixture_tensor, source_tensor, ilens, num_sources


if __name__ == "__main__":
    data = "/ws/ifp-10_3/hasegawa/junzhez2/asteroid/dataset"
    suffixes = [f"{n_src}speakers/wav8k/min" for n_src in [2, 3, 4, 5]]
    tr_json = [os.path.join(data, suffix, "tr") for suffix in suffixes]
    cv_json = [os.path.join(data, suffix, "cv") for suffix in suffixes]
    tt_json = [os.path.join(data, suffix, "tt") for suffix in suffixes]
    dataset_tr = Wsj0mixVariable(tr_json)
    dataloader = torch.utils.data.DataLoader(
        dataset_tr, batch_size=3, collate_fn=_collate_fn, num_workers=3
    )
    print(len(dataset_tr))
    for mixture_tensor, source_tensor, ilens, num_sources in tqdm(dataloader):
        print(mixture_tensor.shape, source_tensor.shape, ilens, num_sources)
