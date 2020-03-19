import torch
from torch.utils import data
import json
import os
import numpy as np
import soundfile as sf

DATASET = 'WHAMR'

# WHAMR tasks
# Many tasks can be considered with this dataset, we only consider the 4 core
# separation tasks presented in the paper for now.
sep_clean = {'mixture': 'mix_clean_anechoic',
             'sources': ['s1_anechoic', 's2_anechoic'],
             'infos': [],
             'default_nsrc': 2}
sep_noisy = {'mixture': 'mix_both_anechoic',
             'sources': ['s1_anechoic', 's2_anechoic'],
             'infos': ['noise'],
             'default_nsrc': 2}
sep_reverb = {'mixture': 'mix_clean_reverb',
              'sources': ['s1_anechoic', 's2_anechoic'],
              'infos': [],
              'default_nsrc': 2}
sep_reverb_noisy = {'mixture': 'mix_both_reverb',
                    'sources': ['s1_anechoic', 's2_anechoic'],
                    'infos': ['noise'],
                    'default_nsrc': 2}

WHAMR_TASKS = {'sep_clean': sep_clean,
               'sep_noisy': sep_noisy,
               'sep_reverb': sep_reverb,
               'sep_reverb_noisy': sep_reverb_noisy}
# Support both order, confusion is easy
WHAMR_TASKS['sep_noisy_reverb'] = WHAMR_TASKS['sep_reverb_noisy']


class WhamRDataset(data.Dataset):
    """ Dataset class for WHAMR source separation and speech enhancement tasks.

    Args:
        json_dir (str): The path to the directory containing the json files.
        task (str): One of ``'sep_clean'``, ``'sep_noisy'``, ``'sep_reverb'``
            or ``'sep_reverb_noisy'``.

            * ``'sep_clean'`` for two-speaker clean (anechoic) source
                separation.
            * ``'sep_noisy'`` for two-speaker noisy (anechoic) source
                separation.
            * ``'sep_reverb'`` for two-speaker clean reverberant
                source separation.
            * ``'sep_reverb_noisy'`` for two-speaker noisy reverberant source
                separation.

        sample_rate (int, optional): The sampling rate of the wav files.
        segment (float, optional): Length of the segments used for training,
            in seconds. If None, use full utterances (e.g. for test).
        nondefault_nsrc (int, optional): Number of sources in the training
            targets.
            If None, defaults to one for enhancement tasks and two for
            separation tasks.
    """
    def __init__(self, json_dir, task, sample_rate=8000, segment=4.0,
                 nondefault_nsrc=None):
        super(WhamRDataset, self).__init__()
        if task not in WHAMR_TASKS.keys():
            raise ValueError('Unexpected task {}, expected one of '
                             '{}'.format(task, WHAMR_TASKS.keys()))
        # Task setting
        self.json_dir = json_dir
        self.task = task
        self.task_dict = WHAMR_TASKS[task]
        self.sample_rate = sample_rate
        self.seg_len = None if segment is None else int(segment * sample_rate)
        if not nondefault_nsrc:
            self.n_src = self.task_dict['default_nsrc']
        else:
            assert nondefault_nsrc >= self.task_dict['default_nsrc']
            self.n_src = nondefault_nsrc
        self.like_test = self.seg_len is None
        # Load json files
        mix_json = os.path.join(json_dir, self.task_dict['mixture'] + '.json')
        sources_json = [os.path.join(json_dir, source + '.json') for
                        source in self.task_dict['sources']]
        with open(mix_json, 'r') as f:
            mix_infos = json.load(f)
        sources_infos = []
        for src_json in sources_json:
            with open(src_json, 'r') as f:
                sources_infos.append(json.load(f))
        # Filter out short utterances only when segment is specified
        orig_len = len(mix_infos)
        drop_utt, drop_len = 0, 0
        if not self.like_test:
            for i in range(len(mix_infos) - 1, -1, -1):  # Go backward
                if mix_infos[i][1] < self.seg_len:
                    drop_utt += 1
                    drop_len += mix_infos[i][1]
                    del mix_infos[i]
                    for src_inf in sources_infos:
                        del src_inf[i]

        print("Drop {} utts({:.2f} h) from {} (shorter than {} samples)".format(
            drop_utt, drop_len/sample_rate/36000, orig_len, self.seg_len))
        self.mix = mix_infos
        # Handle the case n_src > default_nsrc
        while len(sources_infos) < self.n_src:
            sources_infos.append([None for _ in range(len(self.mix))])
        self.sources = sources_infos

    def __add__(self, wham):
        if self.n_src != wham.n_src:
            raise ValueError('Only datasets having the same number of sources'
                             'can be added together. Received '
                             '{} and {}'.format(self.n_src, wham.n_src))
        if self.seg_len != wham.seg_len:
            self.seg_len = min(self.seg_len, wham.seg_len)
            print('Segment length mismatched between the two Dataset'
                  'passed one the smallest to the sum.')
        self.mix = self.mix + wham.mix
        self.sources = [a + b for a, b in zip(self.sources, wham.sources)]

    def __len__(self):
        return len(self.mix)

    def __getitem__(self, idx):
        """ Gets a mixture/sources pair.
        Returns:
            mixture, vstack([source_arrays])
        """
        # Random start
        if self.mix[idx][1] == self.seg_len or self.like_test:
            rand_start = 0
        else:
            rand_start = np.random.randint(0, self.mix[idx][1] - self.seg_len)
        if self.like_test:
            stop = None
        else:
            stop = rand_start + self.seg_len
        # Load mixture
        x, _ = sf.read(self.mix[idx][0], start=rand_start,
                       stop=stop, dtype='float32')
        seg_len = torch.as_tensor([len(x)])
        # Load sources
        source_arrays = []
        for src in self.sources:
            if src[idx] is None:
                # Target is filled with zeros if n_src > default_nsrc
                s = np.zeros((seg_len, ))
            else:
                s, _ = sf.read(src[idx][0], start=rand_start,
                               stop=stop, dtype='float32')
            source_arrays.append(s)
        sources = torch.from_numpy(np.vstack(source_arrays))
        return torch.from_numpy(x), sources
