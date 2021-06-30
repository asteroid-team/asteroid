from asteroid.data import MUSDB18Dataset
import torch
from pathlib import Path

validation_tracks = [
    "Actions - One Minute Smile",
    "Clara Berry And Wooldog - Waltz For My Victims",
    "Johnny Lokke - Promises & Lies",
    "Patrick Talbot - A Reason To Leave",
    "Triviul - Angelsaint",
    "Alexander Ross - Goodbye Bolero",
    "Fergessen - Nos Palpitants",
    "Leaf - Summerghost",
    "Skelpolu - Human Mistakes",
    "Young Griffo - Pennies",
    "ANiMAL - Rockshow",
    "James May - On The Line",
    "Meaxic - Take A Step",
    "Traffic Experiment - Sirens",
]


def load_datasets(parser, args):
    """Loads the specified dataset from commandline arguments

    Returns:
        train_dataset, validation_dataset
    """

    args = parser.parse_args()

    dataset_kwargs = {
        "root": Path(args.train_dir),
    }

    source_augmentations = Compose(
        [globals()["_augment_" + aug] for aug in args.source_augmentations]
    )

    train_dataset = MUSDB18Dataset(
        split="train",
        sources=args.sources,
        targets=args.sources,
        source_augmentations=source_augmentations,
        random_track_mix=True,
        segment=args.seq_dur,
        random_segments=True,
        sample_rate=args.sample_rate,
        samples_per_track=args.samples_per_track,
        **dataset_kwargs,
    )
    train_dataset = filtering_out_valid(train_dataset)

    valid_dataset = MUSDB18Dataset(
        split="train",
        subset=validation_tracks,
        sources=args.sources,
        targets=args.sources,
        segment=None,
        **dataset_kwargs,
    )

    return train_dataset, valid_dataset


def filtering_out_valid(input_dataset):
    """Filtering out validation tracks from input dataset.

    Return:
        input_dataset (w/o validation tracks)
    """
    input_dataset.tracks = [
        tmp
        for tmp in input_dataset.tracks
        if not (str(tmp["path"]).split("/")[-1] in validation_tracks)
    ]

    return input_dataset


class Compose(object):
    """Composes several augmentation transforms.
    Args:
        augmentations: list of augmentations to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, audio):
        for transform in self.transforms:
            audio = transform(audio)
        return audio


def _augment_gain(audio, low=0.25, high=1.25):
    """Applies a random gain to each source between `low` and `high`"""
    gain = low + torch.rand(1) * (high - low)
    return audio * gain


def _augment_channelswap(audio):
    """Randomly swap channels of stereo sources"""
    if audio.shape[0] == 2 and torch.FloatTensor(1).uniform_() < 0.5:
        return torch.flip(audio, [0])

    return audio
