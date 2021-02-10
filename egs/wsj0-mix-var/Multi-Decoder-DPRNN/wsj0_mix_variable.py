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
    def __init__(
        self, json_dirs, n_srcs=[2, 3, 4, 5], sr=8000, seglen=4.0, minlen=2.0
    ):  # segment and cv_maxlen not implemented
        """
        each line of textfile comes in the form of:
            filename1, dB1, filename2, dB2, ...
            args:
                root: folder where dataset/ is located
                json_folders: folders containing json files, **/dataset/#speakers/wav8k/min/tr/**
                sr: sample rate
                seglen: length of each segment in seconds
                minlen: minimum segment length
        """
        if seglen is None:
            self.seg_len = None
            self.min_len = None
        else:
            self.seg_len = int(seglen * sr)
            self.min_len = int(minlen * sr)
        self.like_test = self.seg_len is None
        self.sr = sr
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
        mixtures_list: B x T, torch.Tensor, padded mixtures
        ilens : B, torch.Tensor, length of each mixture before padding
        sources_list: list of B Tensors, each C x T, where C is (variable) number of source audios
    """
    ilens = []  # shape of mixtures
    mixtures = []  # mixtures, same length as longest source in whole batch
    sources_list = []  # padded sources, same length as mixtures
    maxlen = max([len(mixture) for mixture, sources in batch])
    for mixture, sources in batch:  # compute length to pad to
        assert len(mixture) == len(sources[0])
        ilens.append(len(mixture))
        mixtures.append(pad_audio(mixture, maxlen))
        sources = torch.Tensor(
            np.stack([pad_audio(source, maxlen) for source in sources], axis=0)
        ).float()
        sources_list.append(sources)
    mixtures = torch.Tensor(np.stack(mixtures, axis=0)).float()
    ilens = torch.Tensor(np.stack(ilens)).int()
    return mixtures, ilens, sources_list


if __name__ == "__main__":
    data = "/ws/ifp-10_3/hasegawa/junzhez2/asteroid/dataset"
    suffixes = [f"{n_src}speakers/wav8k/min" for n_src in [2, 3, 4, 5]]
    tr_json = [os.path.join(data, suffix, "tr") for suffix in suffixes]
    cv_json = [os.path.join(data, suffix, "cv") for suffix in suffixes]
    tt_json = [os.path.join(data, suffix, "tt") for suffix in suffixes]
    dataset_tr = Wsj0mixVariable(tr_json)
    dataloader = torch.utils.data.DataLoader(dataset_tr, batch_size=3, collate_fn=_collate_fn)
    print(len(dataset_tr))
    for mixtures, ilens, sources_list in tqdm(dataloader):
        print(mixtures.shape, ilens, [len(sources) for sources in sources_list])
