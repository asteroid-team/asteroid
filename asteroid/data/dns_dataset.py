import torch
from torch.utils import data
import json
import os
import soundfile as sf


class DNSDataset(data.Dataset):
    def __init__(self, json_dir):
        super(DNSDataset, self).__init__()
        self.json_dir = json_dir
        with open(os.path.join(json_dir, 'file_infos.json'), 'r') as f:
            self.mix_infos = json.load(f)

        self.wav_ids = list(self.mix_infos.keys())

    def __len__(self):
        return len(self.wav_ids)

    def __getitem__(self, idx):
        """ Gets a mixture/sources pair.
        Returns:
            mixture, vstack([source_arrays])
        """
        utt_info = self.mix_infos[self.wav_ids[idx]]
        # Load mixture
        x = torch.from_numpy(sf.read(utt_info['mix'], dtype='float32')[0])
        # Load clean
        speech = torch.from_numpy(sf.read(utt_info['clean'],
                                          dtype='float32')[0])
        # Load noise
        noise = torch.from_numpy(sf.read(utt_info['noise'],
                                         dtype='float32')[0])
        return x, speech, noise
