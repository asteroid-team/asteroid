from asteroid.data import MUSDB18Dataset
import argparse
import torch
from pathlib import Path
import tqdm


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


if __name__ == "__main__":
    """dataset tests -> these parameters will go into recipes"""
    parser = argparse.ArgumentParser(description="MUSDB18 dataset test")

    parser.add_argument("--root", type=str, help="root path of dataset")

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--seq-dur", type=float, default=6.0, help="Duration of <=0.0 will result in the full audio"
    )

    parser.add_argument(
        "--samples-per-track",
        type=int,
        default=64,
        help="draws a fixed number of samples per track",
    )

    parser.add_argument(
        "--random-track-mix",
        action="store_true",
        default=False,
        help="Apply random track mixing augmentation",
    )
    parser.add_argument(
        "--source-augmentations", type=str, nargs="+", default=["gain", "channelswap"]
    )
    parser.add_argument("--batch-size", type=int, default=16)

    args = parser.parse_args()

    dataset_kwargs = {
        "root": Path(args.root),
    }

    source_augmentations = Compose(
        [globals()["_augment_" + aug] for aug in args.source_augmentations]
    )

    train_dataset = MUSDB18Dataset(
        split="train",
        source_augmentations=source_augmentations,
        random_track_mix=args.random_track_mix,
        segment=args.seq_dur,
        random_segments=True,
        samples_per_track=64,
        **dataset_kwargs,
    )

    # List of MUSDB18 validation tracks as being used in the `musdb` package
    # See https://github.com/sigsep/sigsep-mus-db/blob/master/musdb/configs/mus.yaml#L41
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

    valid_dataset = MUSDB18Dataset(
        split="train", subset=validation_tracks, segment=None, **dataset_kwargs
    )

    test_dataset = MUSDB18Dataset(split="test", subset=None, segment=None, **dataset_kwargs)

    print("Number of train tracks: ", len(train_dataset.tracks))
    print("Number of validation tracks: ", len(valid_dataset.tracks))
    print("Number of test tracks: ", len(test_dataset.tracks))

    print("Number of train samples: ", len(train_dataset))
    print("Number of validation samples: ", len(valid_dataset))
    print("Number of test samples: ", len(test_dataset))

    train_sampler = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    for x, y in tqdm.tqdm(train_sampler):
        pass
