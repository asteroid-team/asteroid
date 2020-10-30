from pathlib import Path
import torch.utils.data
import random
import torch
import tqdm
import soundfile as sf


class MUSDB18Dataset(torch.utils.data.Dataset):
    """MUSDB18 music separation dataset

    The dataset consists of 150 full lengths music tracks (~10h duration) of
    different genres along with their isolated stems:
        `drums`, `bass`, `vocals` and `others`.

    Out-of-the-box, asteroid does only support MUSDB18-HQ which comes as
    uncompressed WAV files. To use the MUSDB18, please convert it to WAV first:

    - MUSDB18 HQ: https://zenodo.org/record/3338373
    - MUSDB18     https://zenodo.org/record/1117372

    .. note::
        The datasets are hosted on Zenodo and require that users
        request access, since the tracks can only be used for academic purposes.
        We manually check this requests.

    This dataset asssumes music tracks in (sub)folders where each folder
    has a fixed number of sources (defaults to 4). For each track, a list
    of `sources` and a common `suffix` can be specified.
    A linear mix is performed on the fly by summing up the sources

    Due to the fact that all tracks comprise the exact same set
    of sources, random track mixing can be used can be used,
    where sources from different tracks are mixed together.

    Folder Structure:
        >>> #train/1/vocals.wav ---------|
        >>> #train/1/drums.wav ----------+--> input (mix), output[target]
        >>> #train/1/bass.wav -----------|
        >>> #train/1/other.wav ---------/

    Args:
        root (str): Root path of dataset
        sources (:obj:`list` of :obj:`str`, optional): List of source names
            that composes the mixture.
            Defaults to MUSDB18 4 stem scenario: `vocals`, `drums`, `bass`, `other`.
        targets (list or None, optional): List of source names to be used as
            targets. If None, a dict with the 4 stems is returned.
             If e.g [`vocals`, `drums`], a tensor with stacked `vocals` and
             `drums` is returned instead of a dict. Defaults to None.
        suffix (str, optional): Filename suffix, defaults to `.wav`.
        split (str, optional): Dataset subfolder, defaults to `train`.
        subset (:obj:`list` of :obj:`str`, optional): Selects a specific of
            list of tracks to be loaded, defaults to `None` (loads all tracks).
        segment (float, optional): Duration of segments in seconds,
            defaults to ``None`` which loads the full-length audio tracks.
        samples_per_track (int, optional):
            Number of samples yielded from each track, can be used to increase
            dataset size, defaults to `1`.
        random_segments (boolean, optional): Enables random offset for track segments.
        random_track_mix boolean: enables mixing of random sources from
            different tracks to assemble mix.
        source_augmentations (:obj:`list` of :obj:`callable`): list of augmentation
            function names, defaults to no-op augmentations (input = output)
        sample_rate (int, optional): Samplerate of files in dataset.

    Attributes:
        root (str): Root path of dataset
        sources (:obj:`list` of :obj:`str`, optional): List of source names.
            Defaults to MUSDB18 4 stem scenario: `vocals`, `drums`, `bass`, `other`.
        suffix (str, optional): Filename suffix, defaults to `.wav`.
        split (str, optional): Dataset subfolder, defaults to `train`.
        subset (:obj:`list` of :obj:`str`, optional): Selects a specific of
            list of tracks to be loaded, defaults to `None` (loads all tracks).
        segment (float, optional): Duration of segments in seconds,
            defaults to ``None`` which loads the full-length audio tracks.
        samples_per_track (int, optional):
            Number of samples yielded from each track, can be used to increase
            dataset size, defaults to `1`.
        random_segments (boolean, optional): Enables random offset for track segments.
        random_track_mix boolean: enables mixing of random sources from
            different tracks to assemble mix.
        source_augmentations (:obj:`list` of :obj:`callable`): list of augmentation
            function names, defaults to no-op augmentations (input = output)
        sample_rate (int, optional): Samplerate of files in dataset.
        tracks (:obj:`list` of :obj:`Dict`): List of track metadata

    References
        "The 2018 Signal Separation Evaluation Campaign" Stoter et al. 2018.
    """

    dataset_name = "MUSDB18"

    def __init__(
        self,
        root,
        sources=["vocals", "bass", "drums", "other"],
        targets=None,
        suffix=".wav",
        split="train",
        subset=None,
        segment=None,
        samples_per_track=1,
        random_segments=False,
        random_track_mix=False,
        source_augmentations=lambda audio: audio,
        sample_rate=44100,
    ):

        self.root = Path(root).expanduser()
        self.split = split
        self.sample_rate = sample_rate
        self.segment = segment
        self.random_track_mix = random_track_mix
        self.random_segments = random_segments
        self.source_augmentations = source_augmentations
        self.sources = sources
        self.targets = targets
        self.suffix = suffix
        self.subset = subset
        self.samples_per_track = samples_per_track
        self.tracks = list(self.get_tracks())
        if not self.tracks:
            raise RuntimeError("No tracks found.")

    def __getitem__(self, index):
        # assemble the mixture of target and interferers
        audio_sources = {}

        # get track_id
        track_id = index // self.samples_per_track
        if self.random_segments:
            start = random.uniform(0, self.tracks[track_id]["min_duration"] - self.segment)
        else:
            start = 0

        # load sources
        for source in self.sources:
            # optionally select a random track for each source
            if self.random_track_mix:
                # load a different track
                track_id = random.choice(range(len(self.tracks)))
                if self.random_segments:
                    start = random.uniform(0, self.tracks[track_id]["min_duration"] - self.segment)

            # loads the full track duration
            start_sample = int(start * self.sample_rate)
            # check if dur is none
            if self.segment:
                # stop in soundfile is calc in samples, not seconds
                stop_sample = start_sample + int(self.segment * self.sample_rate)
            else:
                # set to None for reading complete file
                stop_sample = None

            # load actual audio
            audio, _ = sf.read(
                Path(self.tracks[track_id]["path"] / source).with_suffix(self.suffix),
                always_2d=True,
                start=start_sample,
                stop=stop_sample,
            )
            # convert to torch tensor
            audio = torch.tensor(audio.T, dtype=torch.float)
            # apply source-wise augmentations
            audio = self.source_augmentations(audio)
            audio_sources[source] = audio

        # apply linear mix over source index=0
        audio_mix = torch.stack(list(audio_sources.values())).sum(0)
        if self.targets:
            audio_sources = torch.stack(
                [wav for src, wav in audio_sources.items() if src in self.targets], dim=0
            )
        return audio_mix, audio_sources

    def __len__(self):
        return len(self.tracks) * self.samples_per_track

    def get_tracks(self):
        """Loads input and output tracks"""
        p = Path(self.root, self.split)
        for track_path in tqdm.tqdm(p.iterdir()):
            if track_path.is_dir():
                if self.subset and track_path.stem not in self.subset:
                    # skip this track
                    continue

                source_paths = [track_path / (s + self.suffix) for s in self.sources]
                if not all(sp.exists() for sp in source_paths):
                    print("Exclude track due to non-existing source", track_path)
                    continue

                # get metadata
                infos = list(map(sf.info, source_paths))
                if not all(i.samplerate == self.sample_rate for i in infos):
                    print("Exclude track due to different sample rate ", track_path)
                    continue

                if self.segment is not None:
                    # get minimum duration of track
                    min_duration = min(i.duration for i in infos)
                    if min_duration > self.segment:
                        yield ({"path": track_path, "min_duration": min_duration})
                else:
                    yield ({"path": track_path, "min_duration": None})

    def get_infos(self):
        """Get dataset infos (for publishing models).

        Returns:
            dict, dataset infos with keys `dataset`, `task` and `licences`.
        """
        infos = dict()
        infos["dataset"] = self.dataset_name
        infos["task"] = "enhancement"
        infos["licenses"] = [musdb_license]
        return infos


musdb_license = dict()
