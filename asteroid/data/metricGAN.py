import numpy as np
import soundfile as sf
import torch
from asteroid.data.librimix_dataset import LibriMix
import random as random
from scipy import signal


class MetricGAN(LibriMix):

    def __getitem__(self, idx):
        # Get the row in dataframe
        row = self.df.iloc[idx]
        # Get mixture path
        self.mixture_path = row['mixture_path']
        sources_list = []
        # If there is a seg start point is set randomly
        if self.seg_len is not None:
            start = random.randint(0, row['length'] - self.seg_len)
            stop = start + self.seg_len
        else:
            start = 0
            stop = None
        # If task is enh_both then the source is the clean mixture
        if 'enh_both' in self.task:
            mix_clean_path = self.df_clean.iloc[idx]['mixture_path']
            s, _ = sf.read(mix_clean_path, dtype='float32', start=start,
                           stop=stop)
            sources_list.append(s)

        else:
            # Read sources
            for i in range(self.n_src):
                source_path = row[f'source_{i + 1}_path']
                s, _ = sf.read(source_path, dtype='float32', start=start,
                               stop=stop)
                sources_list.append(s)
        # Read the mixture
        mixture, _ = sf.read(self.mixture_path, dtype='float32', start=start,
                             stop=stop)
        # Convert to torch tensor
        mixture = torch.from_numpy(mixture).unsqueeze(0)
        # Stack sources
        sources = np.vstack(sources_list)
        # Convert sources to tensor
        sources = torch.from_numpy(sources)
        return mixture, sources
