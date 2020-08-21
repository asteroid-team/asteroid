from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import torch.utils.data

import numpy as np
import pandas as pd
import random
import json
import librosa
import warnings
warnings.filterwarnings('ignore')


class DAMPVSEPDataset(torch.utils.data.Dataset):
    """
    DAMP-VSEP vocal separation dataset

    This dataset treats all songs as SINGLE by converting DUETS into 2 different
    performances.

    Note: The datasets are hosted on Zenodo and require that users
          request access.
          DAMP-VSEP https://zenodo.org/record/3553059

    Args:
        root_path (str): Root path of dataset
        csv_dir (str): Path to dataset split in csv format
        task (str): one of ``'enh_vocal'``,``'enh_both'``
            * ``'enh_vocal'`` for vocal enhanced.
            * ``'enh_both'`` for vocal and background separation.
        split (str): one of ``'train'``, ``'valid'`` and ``'test'``
        samples_per_track (int, optional):
            Number of samples yielded from each track, can be used to increase
            dataset size, defaults to `1`.
        random_segments (boolean, optional): Enables random offset for track segments
        sample_rate (int, optional): Samplerate of files in dataset.
            Default 16000 Hz
        segment (float, optional): Duration of segments in seconds,
            Defaults to ``None`` which loads the full-length audio tracks.
        snr (float): SNR between vocal and background. Default 0.
        silence_in_segment (float, optional): Max duration of continuous silence in a segment.
            Default ``None`` which allows any silence length
        num_workers (int, optional): Number of workers.
            Default to ``None`` which utilise all available workers
        norm (Str): Type of normalisation to use
            * ``'song_level'`` use mixture mean and std.
            * ```None``` no normalisation
     """

    dataset_name = 'DAMP-VSEP'

    def __init__(self, root_path, task='enh_both', split=None, samples_per_track=1,
                 random_segments=False, sample_rate=16000,
                 segment=None, snr=0, silence_in_segment=None, num_workers=None, norm=None):

        self.root_path = Path(root_path).expanduser()

        assert task in ['enh_vocal', 'enh_both'], "task should be one of 'enh_vocal','enh_both'"
        self.task = task

        if task == 'enh_vocal':
            self.target = ['vocal']
        elif task == 'enh_both':
            self.target = ['vocal', 'background']

        self.split = split
        self.samples_per_track = samples_per_track
        self.random_segments = random_segments
        self.sample_rate = sample_rate
        self.segment = segment
        self.snr = snr
        self.silence_in_segment = silence_in_segment
        self.num_workers = cpu_count() if num_workers is None else num_workers
        self.tracks = self.get_tracks()
        self.perf_key = [*self.tracks]  # list of performances keys

        self.norm = norm

    def __len__(self):
        return len(self.tracks) * self.samples_per_track

    def __getitem__(self, index):
        audio_sources = {}
        # get track_id

        track_id = index // self.samples_per_track
        perf = self.perf_key[track_id]
        self.mixture_path = perf

        min_duration = int(min(self.tracks[perf]['vocal_duration'],
                               self.tracks[perf]['background_duration']))
        # Set start time of segment
        start = 0
        if self.random_segments:
            start = random.randint(
                0, int(min_duration - self.segment))

        # Set end time of segment
        stop = min_duration
        if self.segment:
            stop = start + self.segment

        for source in ['vocal', 'background']:
            x, _ = librosa.load(
                self.root_path / self.tracks[perf][source],
                sr=self.sample_rate,
                mono=True,
                offset=start,
                duration=stop - start,
                res_type='polyphase'
            )
            audio_sources[source] = x
        
        audio_sources['vocal'] *= float(self.tracks[perf]['scaler'])
            
        if self.norm == 'song_level':
            mix_mean = float(self.tracks[perf]['mean'])
            mix_std = float(self.tracks[perf]['std'])
        else:
            mix_mean = 0.
            mix_std = 1.

        if self.target:
            audio_sources = torch.stack([
                (torch.from_numpy(wav.T) - mix_mean) / mix_std for src, wav in audio_sources.items()
                if src in self.target
            ], dim=0)
        
        audio_mix = audio_sources.sum(0)
        audio_mix = (audio_mix - mix_mean) / mix_std
    
        return audio_mix, audio_sources
    
    def get_track_name(self, idx):
        track_id = idx // self.samples_per_track
        return self.perf_key[track_id]

    def get_tracks(self):
        """
        Loads tracks
        """
        metadata_path = Path(f'metadata/{self.split}_sr{self.sample_rate}.json')
        if metadata_path.exists():
            print(f"Metadata for {self.split} set exist!! Using it.")
            tracks = json.load(open(metadata_path, 'r'))
        else:
            print(f"Constructing Metadata for {self.split} set")
            track_list = pd.read_csv(f"local/{self.split}.csv")
            tracks = self.build_metadata(track_list)

            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            json.dump(tracks, open(metadata_path, 'w'), indent=2)

        return tracks
        
    @staticmethod
    def _build_metadata(inputs):
        sample, root, sample_rate, snr = inputs
        back, _ = librosa.load(root / sample['background_path'],
                               sr=sample_rate, res_type='polyphase')
        back_dur = librosa.get_duration(back, sr=sample_rate)

        vocal, _ = librosa.load(root / sample['vocal_path'],
                                sr=sample_rate, res_type='polyphase')
        vocal_dur = librosa.get_duration(vocal, sr=sample_rate)

        min_dur = min(back_dur, vocal_dur)
        mix = vocal[:int(min_dur * sample_rate)] + back[:int(min_dur * sample_rate)]

        amplitude_scaler = _get_amplitude_scaling_factor(vocal, back, snr=snr)

        track_info = {'mean': f"{mix.mean():.16f}",
                     'std': f"{mix.std():.16f}",
                     'scaler': f"{amplitude_scaler:.16f}",
                     'vocal': sample['vocal_path'],
                     'background': sample['background_path'],
                     'vocal_duration': vocal_dur,
                     'background_duration': back_dur}
        return sample['perf_key'], track_info

    def build_metadata(self, tracks):
        """
        Get the duration using librosa is slower than with soundfile,
        but soundfile can't deal with M4A formats.
        """
        metadata = []
        pool = Pool(processes=self.num_workers)
        track_inputs = [(t, self.root_path, self.sample_rate, self.snr) for i, t in tracks.iterrows()]
        for meta in tqdm(pool.imap_unordered(self._build_metadata, track_inputs), total=len(track_inputs)):
            if meta:
                metadata.append(meta)
        #  return {p: m for p, m in metadata if len(self.get_silence(m, 25000)) == 0}
        return {p: m for p, m in metadata}

    def get_infos(self):
        """ Get dataset infos (for publishing models).

        Returns:
            dict, dataset infos with keys `dataset`, `task` and `licences`.
        """
        infos = dict()
        infos['dataset'] = self.dataset_name
        infos['task'] = self.task
        infos['licenses'] = [dampvsep_license]
        return infos


def _get_amplitude_scaling_factor(v, b, snr=0):
    """Given v and b, return the scaler s according to the snr.
    Args:
      v: ndarray, vocal.
      b: ndarray, backgroun.
      snr: float, SNR. Default=0
    Returns:
      float, scaler.
    """
    def _rms(y):
        return np.sqrt(np.mean(np.abs(y) ** 2, axis=0, keepdims=False))

    v[v == 0.] = 1e-8  # Replace zero values by a small value
    b[b == 0.] = 1e-8  # Replace zero values by a small value
    original_sn_rms_ratio = _rms(v) / _rms(b)
    target_sn_rms_ratio = 10. ** (float(snr) / 20.)
    signal_scaling_factor = target_sn_rms_ratio / original_sn_rms_ratio
    return signal_scaling_factor


dampvsep_license = dict(
    title='DAMP-VSEP: Smule Digital Archive of Mobile Performances - Vocal Separation (Version 1.0.1) ',
    title_link='https://zenodo.org/record/3553059',
    author='Smule, Inc',
    author_link='https://zenodo.org/record/3553059',
    license="Smule's Research Data License Agreement",
    license_link='https://zenodo.org/record/3553059',
    non_commercial=True
)