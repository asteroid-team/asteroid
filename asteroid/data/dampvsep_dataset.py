from pathlib import Path

import torch.utils.data
import random
import json


class DAMPVSEPSinglesDataset(torch.utils.data.Dataset):
    """DAMP-VSEP vocal separation dataset

    This dataset utilises one of the two preprocessed versions of DAMP-VSEP
    from https://github.com/groadabike/DAMP-VSEP-Singles aimed for
    SINGLE SINGER separation.

    The DAMP-VSEP dataset is hosted on Zenodo.
    https://zenodo.org/record/3553059

    Args:
        root_path (str): Root path to DAMP-VSEP dataset.
        task (str): one of ``'enh_vocal'``,``'separation'``.

            * ``'enh_vocal'`` for vocal enhanced.
            * ``'separation'`` for vocal and background separation.

        split (str):  one of ``'train_english'``, ``'train_singles'``,
            ``'valid'`` and ``'test'``.
            Default to ``'train_singles'``.
        ex_per_track (int, optional):
            Number of samples yielded from each track, can be used to increase
            dataset size, defaults to ``1``.
        random_segments (boolean, optional): Enables random offset for track segments.
        sample_rate (int, optional): Sample rate of files in dataset.
            Default 16000 Hz
        segment (float, optional): Duration of segments in seconds,
            Defaults to ``None`` which loads the full-length audio tracks.
        norm (Str, optional): Type of normalisation to use. Default to ``None``

            * ``'song_level'`` use mixture mean and std.
            * ```None``` no normalisation

        source_augmentations (Callable, optional): Augmentations applied to the sources (only).
            Default to ``None``.
        mixture (str, optional): Whether to use the original mixture with non-linear effects
            or remix sources. Default to original.

            * ``'remix'`` for use addition to remix the sources.
            * ``'original'`` for use the original mixture.

    .. note:: There are 2 train set available:

        * train_english: Uses all English spoken song. Duets are converted into 2 singles.
          Totalling 9243 performances and 77Hrs.
        * train_singles: Uses all singles performances, discarding all duets.
          Totalling 20660 performances and 149 hrs.

    """

    dataset_name = "DAMP-VSEP"

    def __init__(
        self,
        root_path,
        task,
        split="train_singles",
        ex_per_track=1,
        random_segments=False,
        sample_rate=16000,
        segment=None,
        norm=None,
        source_augmentations=None,
        mixture="original",
    ):

        self.sample_rate = sample_rate

        self.root_path = Path(root_path).expanduser()
        # Task detail parameters
        assert task in ["enh_vocal", "separation"], "Task should be one of 'enh_vocal','separation'"
        assert mixture in ["remix", "original"], "Mixture should be one of 'remix', 'original'"

        self.task = task
        if task == "enh_vocal":
            self.target = ["vocal"]
        elif task == "separation":
            self.target = ["vocal", "background"]

        self.split = split
        self.tracks = self.get_tracks()
        self.perf_key = [*self.tracks]  # list of performances keys

        self.ex_per_track = ex_per_track
        self.random_segments = random_segments
        self.sample_rate = sample_rate
        self.segment = segment
        self.norm = norm
        self.source_augmentations = source_augmentations
        self.mixture = mixture
        if self.mixture == "original" and self.split == "train_english":
            raise Exception("The 'train_english' train can only accept 'remix' mixture.")

    def __len__(self):
        return len(self.tracks) * self.ex_per_track

    def _load_audio(self, path, start=0.0, duration=None, scaler=None, mean=0.0, std=1.0):
        import librosa

        # ignore warning related with
        # https://github.com/librosa/librosa/issues/1015
        # Soundfile can read OGG (vocal) but not M4A (background and mixture)
        import warnings

        warnings.filterwarnings("ignore", category=UserWarning)

        x, _ = librosa.load(
            path,
            sr=self.sample_rate,
            mono=True,
            offset=start,
            duration=duration,
            dtype="float32",
            res_type="polyphase",
        )
        if scaler:
            x *= scaler
        x -= mean
        x /= std

        return x

    def __getitem__(self, index):
        import math

        audio_sources = {}
        track_id = index // self.ex_per_track
        perf = self.perf_key[track_id]

        self.mixture_path = perf

        # Set start time of segment
        start = 0.0
        duration = math.floor(float(self.tracks[perf]["duration"]) * 100) / 100
        if self.random_segments:
            start = random.uniform(0.0, float(self.tracks[perf]["duration"]) - self.segment)
            duration = float(self.segment)

        mix_mean = 0.0
        mix_std = 1.0
        if self.norm == "song_level":
            if self.mixture == "original":
                mix_mean = float(self.tracks[perf]["original_mix_mean"])
                mix_std = float(self.tracks[perf]["original_mix_std"])
            elif self.mixture == "remix":
                mix_mean = float(self.tracks[perf]["mean"])
                mix_std = float(self.tracks[perf]["std"])

        for source in ["vocal", "background"]:
            scaler = None
            if source == "vocal":
                scaler = float(self.tracks[perf]["scaler"])

            x = self._load_audio(
                self.root_path / self.tracks[perf][source],
                start=start + float(self.tracks[perf][f"{source}_start"]),
                duration=duration,
                scaler=scaler,
                mean=mix_mean,
                std=mix_std,
            )
            if self.source_augmentations:
                x = self.source_augmentations(x, self.sample_rate)
            x = torch.from_numpy(x.T)

            audio_sources[source] = x

        # Prepare targets and mixture
        audio_sources = torch.stack(
            [wav for src, wav in audio_sources.items() if src in self.target], dim=0
        )

        if self.mixture == "remix":
            audio_mix = audio_sources.sum(0)
        else:
            audio_mix = self._load_audio(
                self.root_path / self.tracks[perf]["original_mix"],
                start=start + float(self.tracks[perf]["background_start"]),
                duration=duration,
                mean=mix_mean,
                std=mix_std,
            )

        return audio_mix, audio_sources

    def get_track_name(self, idx):
        track_id = idx // self.ex_per_track
        return self.perf_key[track_id]

    def get_tracks(self):
        """Loads metadata with tracks info.
        Raises error if metadata doesn't exist.
        """
        metadata_path = Path(f"metadata/{self.split}_sr{self.sample_rate}.json")
        if metadata_path.exists():
            tracks = json.load(open(metadata_path, "r"))
        else:
            raise RuntimeError(f"Metadata file for {self.split} not found")
        return tracks

    def get_infos(self):
        """Get dataset infos (for publishing models).

        Returns:
            dict, dataset infos with keys `dataset`, `task` and `licences`.
        """
        infos = dict()
        infos["dataset"] = self.dataset_name
        infos["task"] = self.task
        infos["licenses"] = [dampvsep_license]
        return infos


dampvsep_license = dict(
    title="DAMP-VSEP: Smule Digital Archive of Mobile Performances - Vocal Separation (Version 1.0.1) ",
    title_link="https://zenodo.org/record/3553059",
    author="Smule, Inc",
    author_link="https://zenodo.org/record/3553059",
    license="Smule's Research Data License Agreement",
    license_link="https://zenodo.org/record/3553059",
    non_commercial=True,
)
