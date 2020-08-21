from asteroid.data import DAMPVSEPDataset
import argparse
from pathlib import Path
from tqdm import tqdm

if __name__ == "__main__":
    """dataset tests -> these parameters will go into recipes"""
    parser = argparse.ArgumentParser(description='Dataset dataset test')

    parser.add_argument(
        '--root', type=str, help='root path of dataset'
    )

    parser.add_argument('--batch-size', type=int, default=16)

    args = parser.parse_args()

    dataset_kwargs = {
        'root_path': Path(args.root),
        'task': 'enh_both'
    }

    train_set = DAMPVSEPDataset(
        split='train',
        segment=10,
        samples_per_track=2,
        **dataset_kwargs
    )

    print("Number of train tracks: ", len(train_set.tracks))
    print("Number of train samples: ", len(train_set))

    valid_dataset = DAMPVSEPDataset(
        split='valid',
        **dataset_kwargs
    )
    print("Number of valid tracks: ", len(valid_dataset.tracks))
    print("Number of valid samples: ", len(valid_dataset))

    test_dataset = DAMPVSEPDataset(
        split='test',
        **dataset_kwargs
    )
    print("Number of test tracks: ", len(test_dataset.tracks))
    print("Number of test samples: ", len(test_dataset))

    for x, y in tqdm(test_dataset):
        pass

