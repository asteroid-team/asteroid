from pathlib import Path
import torch.utils.data
import random
import torch
import tqdm
import soundfile as sf


class MUSDB18Dataset(torch.utils.data.Dataset):
    """A dataset of that assumes audio sources to be stored
    in track (subb)folder where each folder has a fixed number of sources.
    For each track the users specifies a list of `source_files` and
    a common `suffix`.
    A linear mix is performed on the fly by summing the target and
    the inferers up.

    Due to the fact that all tracks comprise the exact same set
    of sources, random track mixing augmentation technique
    can be used, where sources from different tracks are mixed
    together.

    Example
    =======
    train/1/vocals.wav ---------------\
    train/1/drums.wav -----------------+--> input (mix), output[target]
    train/1/bass.wav ------------------|
    train/1/other.wav ----------------/

    """
    def __init__(self,
                 root,
                 split='train',
                 subset=None,
                 source_files=['vocals', 'bass', 'drums', 'other'],
                 suffix='.wav',
                 seq_duration=None,
                 samples_per_track=1,
                 random_chunks=False,
                 random_track_mix=False,
                 source_augmentations=lambda audio: audio,
                 sample_rate=44100):

        """MUSDB18 torch.data.Dataset that samples from the MUSDB18 tracks
        using track and excerpts with replacement.

        Parameters
        ----------
        root : str
            root path of MUSDB18HQ, please download from zenodo
        suffix : str
            specify the filename suffice, defaults to `.wav`
        split : str
            select dataset subfolder, defaults to ``train``.
        subsets : list(str)
            selects a list of track folders, defaults to `None` 
            which loads all tracks
        seq_duration : float
            training is performed in chunks of ``seq_duration`` (in seconds,
            defaults to ``None`` which loads the full audio track
        samples_per_track : int
            sets the number of samples, yielded from each track
            Defaults to 1
        source_augmentations : list[callables]
            provide list of augmentation function that take a multi-channel
            audio file of shape (src, samples) as input and output. Defaults to
            no-augmentations (input = output)
        random_track_mix : boolean
            randomly mixes sources from different tracks to assemble a
            custom mix. This augmenation is only applied for the train subset.
        args, kwargs : additional keyword arguments
            used to add further control for the musdb dataset
            initialization function.
        """
        self.root = Path(root).expanduser()
        self.split = split
        self.sample_rate = sample_rate
        self.seq_duration = seq_duration
        self.random_track_mix = random_track_mix
        self.random_chunks = random_chunks
        self.source_augmentations = source_augmentations
        self.source_files = source_files
        self.suffix = suffix
        self.subset = subset
        self.samples_per_track = samples_per_track
        self.tracks = list(self.get_tracks())
        if not self.tracks:
            raise RuntimeError("No tracks found")

    def __getitem__(self, index):
        # assemble the mixture of target and interferers
        audio_sources = {}
        # load interferers
        for source in self.source_files:
            # optionally select a random track for each source
            if self.random_track_mix:
                track_id = random.choice(range(len(self.tracks)))
            else:
                track_id = index // self.samples_per_track

            track_path = self.tracks[track_id]['path']
            if self.random_chunks:
                min_duration = self.tracks[track_id]['min_duration']
                start = random.uniform(0, min_duration - self.seq_duration)

            # loads the full track duration
            start = int(start * self.sample_rate)
            # check if dur is none
            if self.seq_duration:
                # stop in soundfile is calc in samples, not seconds
                stop = start + int(self.seq_duration * self.sample_rate)
            else:
                # set to None for reading complete file
                stop = None

            audio, _ = sf.read(
                Path(track_path / source).with_suffix(self.suffix),
                always_2d=True,
                start=start,
                stop=stop
            )
            audio = torch.tensor(audio.T, dtype=torch.float)
            audio = self.source_augmentations(audio)
            audio_sources[source] = audio

        # apply linear mix over source index=0
        audio_mix = torch.stack(list(audio_sources.values())).sum(0)
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

                source_paths = [
                    track_path / (s + self.suffix) for s in self.source_files
                ]
                if not all(sp.exists() for sp in source_paths):
                    print(
                        "Exclude track due to non-existing source",
                        track_path
                    )
                    continue

                # get metadata
                infos = list(map(sf.info, source_paths))
                if not all(
                    i.samplerate == self.sample_rate for i in infos
                ):
                    print(
                        "Exclude track due to different sample rate ",
                        track_path
                    )
                    continue

                if self.seq_duration is not None:
                    # get minimum duration of track
                    min_duration = min(i.duration for i in infos)
                    if min_duration > self.seq_duration:
                        yield({
                            'path': track_path,
                            'min_duration': min_duration
                        })
                else:
                    yield({
                        'path': track_path,
                        'min_duration': None
                    })
