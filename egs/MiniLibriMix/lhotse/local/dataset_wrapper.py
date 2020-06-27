from torch.utils.data import Dataset
import numpy as np
import torch
import os
import lilcom
import yaml
from lhotse.utils import load_yaml


def parse_yaml(y):
    y = load_yaml(y)

    rec_ids = {}
    for entry in y:
        key = entry["features"]["recording_id"]
        if key not in rec_ids.keys():
            rec_ids[key] = [entry["features"]["storage_path"]]
        else:
            rec_ids[key].append(entry["features"]["storage_path"])

    return rec_ids

class OnTheFlyMixing(Dataset):

    def __init__(self, sources_yaml = './data/cuts_sources.yml.gz', mixture_yaml = './data/cuts_mix.yml.gz',
                 noise_yaml = './data/cuts_noise.yml.gz',
        ):
        self.sources = parse_yaml(sources_yaml)
        self.mixtures = parse_yaml(mixture_yaml) # not used
        self.noises = parse_yaml(noise_yaml)
        self.ids = [k for k in self.mixtures.keys()]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):

        k = self.ids[item]
        c_sources = []
        for s in self.sources[k]:
            with open(s, 'rb') as f:
                c_sources.append(lilcom.decompress(f.read()))
        c_sources = np.stack(c_sources)

        with open(self.noises[k][0], 'rb') as f:
            c_noise = lilcom.decompress(f.read())

        #with open(self.mixtures[k][0], 'rb') as f:
           # c_mix = lilcom.decompress(f.read())
        c_sources = torch.exp(torch.from_numpy(c_sources).float())
        onthefly = torch.sum(c_sources, 0) + torch.exp(torch.from_numpy(c_noise).float())

        return {"mixture": torch.log(onthefly), "sources": torch.log(c_sources)}


class LhotseDataset(Dataset):
    def __init__(self, dataset, target_length, frames_dim=0):
        self.dataset = dataset # dataset which return feats of unequal length.
        self.target_length = target_length # target length (samples or frames)
        self.frames_dim = frames_dim # tensor dimension for sequence length.

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        # if longer than self.target_length --> we take a random chunk
        # if shorter we may want to pad it. In feature domain it makes sense (we have losses mainly on a per-frame
        # basis, in time domain no).
        out = self.dataset[item]
        # we iterate over the outputs and select random chunks
        for i, k in enumerate(out.keys()):
            if k in ["real_mask", "binary_mask"]:
                continue
            tmp = out[k]
            frames_dim = self.frames_dim if len(tmp.shape) == 2 else self.frames_dim + 1 # handle sources
            if tmp.shape[frames_dim] < self.target_length:
                raise NotImplementedError # TODO
            elif tmp.shape[frames_dim] > self.target_length:
                # we chunk
                if i == 0:
                    offset = np.random.randint(0, tmp.shape[frames_dim] - self.target_length)
                # offset is the same for sources and mixture
                tmp = tmp.narrow(dim=frames_dim,
                                       start=offset, length= self.target_length)
            out[k] = tmp

        return torch.exp(out["mixture"]).transpose(0, -1), torch.exp(out["sources"]).transpose(1, -1)