import torch
from torch.utils import data
import numpy as np
import soundfile as sf
from asteroid.data.wham_dataset import normalize_tensor_wav

from .wsj0_mix import wsj0_license

EPS = 1e-8

DATASET = "SMS_WSJ"
# SMS_WSJ targets
sep_source = {
    "mixture": "observation",
    "target": ["speech_source"],
    "infos": {"num_channels": 6},
    "default_nsrc": 2,
}
sep_early = {
    "mixture": "observation",
    "target": ["speech_reverberation_early"],
    "infos": {"num_channels": 6},
    "default_nsrc": 2,
}

# speech image represents the whole reverberated signal so it is the sum of
# the early and tail reverberation.
speech_image = ["speech_reverberation_early", "speech_reverberation_tail"]
sep_image = {
    "mixture": "observation",
    "target": speech_image,
    "infos": {"num_channels": 6},
    "default_nsrc": 2,
}


SMS_TARGETS = {"source": sep_source, "early": sep_early, "image": sep_image}


class SmsWsjDataset(data.Dataset):
    """ Dataset class for SMS WSJ source separation.

    Args:
        json_path (str): The path to the sms_wsj json file.
        target (str): One of ``'source'``, ``'early'`` or ``'image'``.

            * ``'source'`` non reverberant clean targets signals.
            * ``'early'`` early reverberation target signals.
            * ``'image'`` reverberant target signals
        dset (str): train_si284 for train, cv_dev93 for validation and
                test_eval92 for test
        sample_rate (int, optional): The sampling rate of the wav files.
        segment (float, optional): Length of the segments used for training,
            in seconds. If None, use full utterances (e.g. for test).
        single_channel (bool): if False all channels are used if True only
                    a random channel is used during training
                    and the first channel during test
        nondefault_nsrc (int, optional): Number of sources in the training
            targets.
        normalize_audio (bool): If True then both sources and the mixture are
            normalized with the standard deviation of the mixture.

    References:
        "SMS-WSJ: Database, performance measures, and baseline recipe for
         multi-channel source separation and recognition", Drude et al. 2019
    """

    dataset_name = "SMS_WSJ"

    def __init__(
        self,
        json_path,
        target,
        dset,
        sample_rate=8000,
        single_channel=True,
        segment=4.0,
        nondefault_nsrc=None,
        normalize_audio=False,
    ):
        try:
            import sms_wsj  # noqa
        except ModuleNotFoundError:
            import warnings

            warnings.warn(
                "Some of the functionality relies on the sms_wsj package "
                "downloadable from https://github.com/fgnt/sms_wsj ."
                "The user is encouraged to install the package"
            )
        super().__init__()
        if target not in SMS_TARGETS.keys():
            raise ValueError(
                "Unexpected task {}, expected one of " "{}".format(target, SMS_TARGETS.keys())
            )

        # Task setting
        self.json_path = json_path
        self.target = target
        self.target_dict = SMS_TARGETS[target]
        self.single_channel = single_channel
        self.sample_rate = sample_rate
        self.normalize_audio = normalize_audio
        self.seg_len = None if segment is None else int(segment * sample_rate)
        if not nondefault_nsrc:
            self.n_src = self.target_dict["default_nsrc"]
        else:
            assert nondefault_nsrc >= self.target_dict["default_nsrc"]
            self.n_src = nondefault_nsrc
        self.like_test = self.seg_len is None
        self.dset = dset

        # Load json files

        from lazy_dataset.database import JsonDatabase

        db = JsonDatabase(json_path)
        dataset = db.get_dataset(dset)
        # Filter out short utterances only when segment is specified
        if not self.like_test:

            def filter_short_examples(example):
                num_samples = example["num_samples"]["observation"]
                if num_samples < self.seg_len:
                    return False
                else:
                    return True

            dataset = dataset.filter(filter_short_examples, lazy=False)
        self.dataset = dataset

    def __add__(self, sms_wsj):
        if self.n_src != sms_wsj.n_src:
            raise ValueError(
                "Only datasets having the same number of sources"
                "can be added together. Received "
                "{} and {}".format(self.n_src, sms_wsj.n_src)
            )
        if self.seg_len != sms_wsj.seg_len:
            self.seg_len = min(self.seg_len, sms_wsj.seg_len)
            print(
                "Segment length mismatched between the two Dataset"
                "passed one the smallest to the sum."
            )
        self.dataset = self.dataset.concatenate(sms_wsj.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """ Gets a mixture/sources pair.
        Returns:
            mixture, vstack([source_arrays])
        """
        # Random start
        example = self.dataset[idx]
        in_signal = self.target_dict["mixture"]
        target = self.target_dict["target"]
        audio_path = example["audio_path"]
        num_samples = example["num_samples"]["observation"]
        if num_samples == self.seg_len or self.like_test:
            rand_start = 0
        else:
            rand_start = np.random.randint(0, num_samples - self.seg_len)
        if self.like_test:
            stop = None
        else:
            stop = rand_start + self.seg_len
        # Load mixture
        x, _ = sf.read(audio_path[in_signal], start=rand_start, stop=stop, dtype="float32")
        x = x.T

        num_channels = self.target_dict["infos"]["num_channels"]
        if self.single_channel:
            if self.like_test:
                ref_channel = 0
            else:
                ref_channel = np.random.randint(0, num_channels)
            x = x[ref_channel]
        seg_len = torch.as_tensor([x.shape[-1]])
        # Load sources
        source_arrays = []
        for idx in range(self.n_src):
            try:
                s = 0
                for t in target:
                    if t == "speech_source":
                        start = 0
                        stop_ = None
                    else:
                        start = rand_start
                        stop_ = stop
                    s_, _ = sf.read(audio_path[t][idx], start=start, stop=stop_, dtype="float32")
                    s += s_.T
            except IndexError:
                if self.single_channel:
                    s = np.zeros((seg_len,))
                else:
                    s = np.zeros((num_channels, seg_len))
            source_arrays.append(s)

        if target[0] == "speech_source":
            from sms_wsj.database.utils import extract_piece

            offset = example["offset"]
            source_arrays = [
                extract_piece(s, offset_, num_samples) for s, offset_ in zip(source_arrays, offset)
            ]
            source_arrays = [s[rand_start:stop] for s in source_arrays]

        sources = torch.from_numpy(np.stack(source_arrays, axis=0))
        assert sources.shape[-1] == seg_len[0], (sources.shape, seg_len)
        if self.single_channel and not target[0] == "source":
            sources = sources[:, ref_channel]

        mixture = torch.from_numpy(x)

        if self.normalize_audio:
            m_std = mixture.std(-1, keepdim=True)
            mixture = normalize_tensor_wav(mixture, eps=EPS, std=m_std)
            sources = normalize_tensor_wav(sources, eps=EPS, std=m_std)
        return mixture, sources

    def get_infos(self):
        """ Get dataset infos (for publishing models).

        Returns:
            dict, dataset infos with keys `dataset`, `task` and `target`.
        """
        infos = dict()
        infos["dataset"] = self.dataset_name
        infos["task_dataset"] = self.dset
        infos["target"] = self.target
        infos["licenses"] = [wsj0_license, wsj1_license, sms_wsj_license]
        return infos


wsj1_license = dict(
    title="CSR-II (WSJ1) Complete",
    title_link="https://catalog.ldc.upenn.edu/LDC94S13A",
    author="LDC",
    author_link="https://www.ldc.upenn.edu/",
    license="LDC User Agreement for Non-Members",
    license_link="https://catalog.ldc.upenn.edu/license/ldc-non-members-agreement.pdf",
    non_commercial=True,
)


sms_wsj_license = dict(
    title="SMS-WSJ: A database for in-depth analysis of multi-channel source separation algorithms",
    title_link="https://github.com/fgnt/sms_wsj",
    author="Department of Communications Engineering University of Paderborn",
    author_link="https://github.com/fgnt",
    license="MIT License",
    license_link="https://github.com/fgnt/sms_wsj/blob/master/LICENSE",
    non_commercial=False,
)
