import json
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
import random


class LibriVADDataset(Dataset):
    """Dataset class for Voice Activity Detection.

    Args:
        md_file_path (str): The path to the metadata file.
    """

    def __init__(self, md_file_path, sample_rate=8000, segment=3):

        self.md_filepath = md_file_path
        with open(self.md_filepath) as json_file:
            self.md = json.load(json_file)
        self.segment = segment
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.md)

    def __getitem__(self, idx):
        # Get the row in dataframe
        row = self.md[idx]
        # Get mixture path
        self.source_path = row[f"mixture_path"]
        length = len(sf.read(self.source_path)[0])
        if self.segment is not None:
            start = random.randint(0, length - int(self.segment * self.sample_rate))
            stop = start + int(self.segment * self.sample_rate)
        else:
            start = 0
            stop = None

        s, sr = sf.read(self.source_path, start=start, stop=stop, dtype="float32")
        # Convert sources to tensor
        source = torch.from_numpy(s)
        label = from_vad_to_label(length, row["VAD"], start, stop).unsqueeze(0)
        return source, label


def from_vad_to_label(length, vad, begin, end):
    label = torch.zeros(length, dtype=torch.float)
    for start, stop in zip(vad["start"], vad["stop"]):
        label[..., start:stop] = 1
    return label[..., begin:end]
