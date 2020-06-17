import numpy as np
import soundfile as sf
import torch
from asteroid.data.librimix_dataset import LibriMix
import random as random
from scipy import signal


class SEGAN(LibriMix):

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
        mixture = self.pre_emphasis(mixture).astype('float32')
        # Convert to torch tensor
        mixture = torch.from_numpy(mixture).unsqueeze(0)
        # Stack sources
        sources = np.vstack(sources_list)
        sources = self.pre_emphasis(sources).astype('float32')
        # Convert sources to tensor
        sources = torch.from_numpy(sources)
        if self.segment is None or self.seg_len > 16384 :
            return self.slicer(mixture), sources
        return mixture, sources

    def slicer(self,sources, window=16384):
        len_s = len(sources[0, :])
        if len_s > window:
            nb_slices = int(len_s // window) + 1
            sliced = torch.zeros((sources.size()[0], nb_slices * window))
            sliced = sliced.reshape((sources.size()[0], nb_slices, window))
            for nb_source in range(sources.size()[0]):
                for j in range(nb_slices - 1):
                    sliced[nb_source, j, :] = sources[nb_source,
                                              j * window: (j + 1) * window]
                sliced[nb_source, -1, : len_s - (j + 1) * window] = sources[
                                                                    nb_source,
                                                                    (
                                                                                j + 1) * window:]
            return sliced
        return sources.unsqueeze(1)

    def pre_emphasis(self,signal_batch, emph_coeff=0.95) -> np.array:
        """
        Pre-emphasis of higher frequencies given a batch of signal.

        Args:
            signal_batch(np.array): batch of signals, represented as numpy arrays
            emph_coeff(float): emphasis coefficient

        Returns:
            result: pre-emphasized signal batch
        """
        return signal.lfilter([1, -emph_coeff], [1], signal_batch)