from torch.utils.data import Dataset
import numpy as np

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
        for k in out.keys():
            if k in ["real_mask", "binary_mask"]:
                continue
            tmp = out[k]
            frames_dim = self.frames_dim if len(tmp.shape) == 2 else self.frames_dim + 1 # handle sources
            if tmp.shape[frames_dim] < self.target_length:
                raise NotImplementedError # TODO
            elif tmp.shape[frames_dim] > self.target_length:
                # we chunk
                offset = np.random.randint(0, tmp.shape[frames_dim] - self.target_length)
                tmp = tmp.narrow(dim=frames_dim,
                                       start=offset, length= self.target_length)
            out[k] = tmp
        return out["mixture"].transpose(0, -1), out["sources"].transpose(1, -1)