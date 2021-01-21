import torch
from torch.utils import data
import json
import os
import numpy as np
import soundfile as sf


def make_dataloaders(
    train_dir,
    valid_dir,
    n_src=2,
    sample_rate=8000,
    segment=4.0,
    batch_size=4,
    num_workers=None,
    **kwargs,
):
    num_workers = num_workers if num_workers else batch_size
    train_set = Wsj0mixDataset(train_dir, n_src=n_src, sample_rate=sample_rate, segment=segment)
    val_set = Wsj0mixDataset(valid_dir, n_src=n_src, sample_rate=sample_rate, segment=segment)
    train_loader = data.DataLoader(
        train_set, shuffle=True, batch_size=batch_size, num_workers=num_workers, drop_last=True
    )
    val_loader = data.DataLoader(
        val_set, shuffle=True, batch_size=batch_size, num_workers=num_workers, drop_last=True
    )
    return train_loader, val_loader


class VariableWsj0mixDataset(data.Dataset):
    """
    each line of textfile comes in the form of:
        filename1, dB1, filename2, dB2, ...
    args:
        root: folder where dataset/ is located
        json_folders: folders containing json files, [/dataset/#speakers/wav8k/min/tr/**]
        sample_rate: sample rate
        seglen: length of each segment in seconds; if None, use full utterance
        minlen: minimum segment length
    """

    dataset_name = "wsj0-mix-variable"

    def __init__(
        self, root, json_folders, sample_rate=8000, seglen=4.0, minlen=2.0, debug=False
    ):  # segment and cv_maxlen not implemented
        str_tmp = "_debug" if debug else ""
        seglen = int(seglen * sr)
        minlen = int(minlen * sr)
        self.sr = sample_rate
        self.mixes = []
        for json_folder in json_folders:
            mixfiles, wavlens = list(
                zip(*load_json(os.path.join(root + str_tmp, json_folder, "mix.json")))
            )  # list of 20000 filenames, and 20000 lengths
            mixfiles = [os.path.join(root, mixfile.split("dataset/")[1]) for mixfile in mixfiles]
            sig_json = [
                load_json(file)
                for file in sorted(glob.glob(os.path.join(root + str_tmp, json_folder, "s*.json")))
            ]  # list C, each have 20000 filenames
            for i, spkr_json in enumerate(sig_json):
                sig_json[i] = [
                    os.path.join(root, line[0].split("dataset/")[1]) for line in spkr_json
                ]  # list C, each have 20000 filenames
            siglists = list(zip(*sig_json))  # list of 20000, each have C filenames
            self.mixes += list(zip(mixfiles, siglists, wavlens))
        # printlist(self.mixes)
        self.examples = []
        for i, mix in enumerate(self.mixes):
            if mix[2] < minlen:
                continue
            start = 0
            while start + minlen <= mix[2]:
                end = min(start + seglen, mix[2])
                self.examples.append(
                    {"mixfile": mix[0], "sourcefiles": mix[1], "start": start, "end": end}
                )
                start += minlen
        random.seed(0)
        self.examples = random.sample(self.examples, len(self.examples))

        # Count.
        example_source_files_len = [len(tmp["sourcefiles"]) for tmp in self.examples]
        unique, counts = np.unique(np.array(example_source_files_len), return_counts=True)
        self.example_weights = []
        for tmp in example_source_files_len:
            self.example_weights.append(1.0 / counts[tmp - 2])
        self.example_weights = torch.Tensor(self.example_weights)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        """
        Returns:
            mixture: [T]
            sources: list of C, each [T]
        """
        example = self.examples[idx]
        mixfile, sourcefiles, start, end = (
            example["mixfile"],
            example["sourcefiles"],
            example["start"],
            example["end"],
        )
        mixture, sr = load(mixfile, sr=self.sr)
        assert sr == self.sr, "need to resample"
        mixture = mixture[start:end]
        sources = [load(sourcefile, sr=sr)[0][start:end] for sourcefile in sourcefiles]
        return mixture, sources


def _collate_fn(batch):
    """
    Args:
        batch: list, len(batch) = batch_size, each entry is a tuple of (mixture, sources)
    Returns:
        mixtures_list: B x T, torch.Tensor, padded mixtures
        ilens : B, torch.Tensor, length of each mixture before padding
        sources_list: list of B Tensors, each C x T, where C is (variable) number of source audios
    """
    ilens = []  # shape of mixtures
    mixtures = []  # mixtures, same length as longest source in whole batch
    sources_list = []  # padded sources, same length as mixtures
    for mixture, sources in batch:  # compute length to pad to
        assert len(mixture) == len(sources[0])
        assert len(mixture) <= 32000
        ilens.append(len(mixture))
        mixtures.append(pad_audio(mixture))
        sources = torch.Tensor(np.stack([pad_audio(source) for source in sources], axis=0)).float()
        sources_list.append(sources)
    mixtures = torch.Tensor(np.stack(mixtures, axis=0)).float()
    ilens = torch.Tensor(np.stack(ilens)).int()
    return mixtures, ilens, sources_list


wsj0_license = dict(
    title="CSR-I (WSJ0) Complete",
    title_link="https://catalog.ldc.upenn.edu/LDC93S6A",
    author="LDC",
    author_link="https://www.ldc.upenn.edu/",
    license="LDC User Agreement for Non-Members",
    license_link="https://catalog.ldc.upenn.edu/license/ldc-non-members-agreement.pdf",
    non_commercial=True,
)
