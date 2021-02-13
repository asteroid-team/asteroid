import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from asteroid.data.wham_dataset import WHAM_TASKS
import soundfile as sf
import random
from pysndfx import AudioEffectsChain
import json


class AugmentedWhamDataset(Dataset):
    """Dataset class for WHAM source separation and speech enhancement tasks.

    Args:
        wsj_train_dir (str): The path to the directory containing the wsj train/dev/test .wav files.
        task (str): One of ``'enh_single'``, ``'enh_both'``, ``'sep_clean'`` or
            ``'sep_noisy'``.

            * ``'enh_single'`` for single speaker speech enhancement.
            * ``'enh_both'`` for multi speaker speech enhancement.
            * ``'sep_clean'`` for two-speaker clean source separation.
            * ``'sep_noisy'`` for two-speaker noisy source separation.
        noise_dir (str, optional): The path to the directory containing the WHAM train/dev/test .wav files
        sample_rate (int, optional): The sampling rate of the wav files.
        json_dir: (str, optional):
        segment (float, optional): Length of the segments used for training,
            in seconds. If None, use full utterances (e.g. for test).
        nondefault_nsrc (int, optional): Number of sources in the training
            targets.
            If None, defaults to one for enhancement tasks and two for
            separation tasks.
        global_db_range: (tuple, optional): Minimum and maximum bounds for each source (and noise) (dB).
        global_stats: (tuple, optional): Mean and standard deviation for level in dB of first source.
        rel_stats: (tuple, optional): Mean and standard deviation for level in dB of second source relative to the first source.
        noise_stats: (tuple, optional): Mean and standard deviation for level in dB of noise relative to the first source.
        speed_perturb: (tuple, optional): Range for SoX speed perturbation transformation.
    """

    def __init__(
        self,
        wsj0train,
        task,
        noise_dir=None,
        json_dir=None,
        orig_percentage=0.0,
        sample_rate=8000,
        segment=4.0,
        nondefault_nsrc=None,
        global_db_range=(-45, 0),
        abs_stats=(-16.7, 7),
        rel_stats=(2.52, 4),
        noise_stats=(5.1, 6.4),
        speed_perturb=(0.95, 1.05),
    ):
        super(AugmentedWhamDataset, self).__init__()
        if task not in WHAM_TASKS.keys():
            raise ValueError(
                "Unexpected task {}, expected one of " "{}".format(task, WHAM_TASKS.keys())
            )
        # Task setting
        self.task = task
        if self.task in ["sep_noisy", "enh_single"] and not noise_dir:
            raise RuntimeError(
                "noise directory must be specified if task is sep_noisy or enh_single"
            )
        self.task_dict = WHAM_TASKS[task]
        self.orig_percentage = orig_percentage
        if json_dir:
            self.use_original = True
        else:
            self.use_original = False
        self.sample_rate = sample_rate
        self.seg_len = None if segment is None else int(segment * sample_rate)

        self.global_db_range = global_db_range
        self.abs_stats = abs_stats
        self.rel_stats = rel_stats
        self.noise_stats = noise_stats
        self.speed_perturb = speed_perturb

        if not nondefault_nsrc:
            self.n_src = self.task_dict["default_nsrc"]
        else:
            assert nondefault_nsrc >= self.task_dict["default_nsrc"]
            self.n_src = nondefault_nsrc
        if json_dir:
            self.wham_mix, self.wham_sources = self.parse_wham(json_dir)
        self.hashtab_synth = self.parse_wsj0(wsj0train, noise_dir)

    def parse_wham(self, json_dir):
        mix_json = os.path.join(json_dir, self.task_dict["mixture"] + ".json")
        sources_json = [
            os.path.join(json_dir, source + ".json") for source in self.task_dict["sources"]
        ]
        with open(mix_json, "r") as f:
            mix_infos = json.load(f)
        sources_infos = []
        for src_json in sources_json:
            with open(src_json, "r") as f:
                sources_infos.append(json.load(f))
        # Filter out short utterances only when segment is specified
        orig_len = len(mix_infos)
        drop_utt, drop_len = 0, 0

        for i in range(len(mix_infos) - 1, -1, -1):  # Go backward
            if mix_infos[i][1] < self.seg_len:
                drop_utt += 1
                drop_len += mix_infos[i][1]
                del mix_infos[i]
                for src_inf in sources_infos:
                    del src_inf[i]

        print(
            "Drop {} utts({:.2f} h) from {} (shorter than {} samples)".format(
                drop_utt, drop_len / self.sample_rate / 36000, orig_len, self.seg_len
            )
        )
        mix = mix_infos
        # Handle the case n_src > default_nsrc
        while len(sources_infos) < self.n_src:
            sources_infos.append([None for _ in range(len(mix))])
        sources = sources_infos
        return mix, sources

    def parse_wsj0(self, wsj_train_dir, noise_dir):

        # Load json files
        utterances = glob.glob(os.path.join(wsj_train_dir, "**/*.wav"), recursive=True)
        noises = None
        if self.task in ["sep_noisy", "enh_single", "enhance_single", "enh_both"]:
            noises = glob.glob(os.path.join(noise_dir, "*.wav"))
            assert len(noises) > 0, "No noises parsed. Wrong path?"

        # parse utterances according to speaker
        drop_utt, drop_len = 0, 0
        print("Parsing WSJ speakers")
        examples_hashtab = {}
        for utt in utterances:
            # exclude if too short
            meta = sf.SoundFile(utt)
            c_len = len(meta)
            assert meta.samplerate == self.sample_rate

            target_length = (
                int(np.ceil(self.speed_perturb[1] * self.seg_len))
                if self.speed_perturb
                else self.seg_len
            )
            if c_len < target_length:  # speed perturbation
                drop_utt += 1
                drop_len += c_len
                continue
            speaker = utt.split("/")[-2]
            if speaker not in examples_hashtab.keys():
                examples_hashtab[speaker] = [(utt, c_len)]
            else:
                examples_hashtab[speaker].append((utt, c_len))

        print(
            "Drop {} utts({:.2f} h) from {} (shorter than {} samples)".format(
                drop_utt, drop_len / self.sample_rate / 36000, len(utterances), self.seg_len
            )
        )

        drop_utt, drop_len = 0, 0
        if noises:
            examples_hashtab["noise"] = []
            for noise in noises:
                meta = sf.SoundFile(noise)
                c_len = len(meta)
                assert meta.samplerate == self.sample_rate
                target_length = (
                    int(np.ceil(self.speed_perturb[1] * self.seg_len))
                    if self.speed_perturb
                    else self.seg_len
                )
                if c_len < target_length:  # speed perturbation
                    drop_utt += 1
                    drop_len += c_len
                    continue
                examples_hashtab["noise"].append((noise, c_len))

            print(
                "Drop {} noises({:.2f} h) from {} (shorter than {} samples)".format(
                    drop_utt, drop_len / self.sample_rate / 36000, len(noises), self.seg_len
                )
            )

        return examples_hashtab

    def __add__(self, wham):
        raise NotImplementedError  # It will require different handling of other datasets, I suggest using dicts

    def __len__(self):
        if self.use_original:
            return len(
                self.wham_mix
            )  # same length as original wham (actually if orig_percentage = 1 the data is original wham)
        else:
            return sum(
                [len(self.hashtab_synth[x]) for x in self.hashtab_synth.keys()]
            )  # we account only the wsj0 length

    def random_data_augmentation(self, signal, c_gain, speed):
        if self.speed_perturb:
            fx = (
                AudioEffectsChain().speed(speed).custom("norm {}".format(c_gain))
            )  # speed perturb and then apply gain
        else:
            fx = AudioEffectsChain().custom("norm {}".format(c_gain))
        signal = fx(signal)

        return signal

    @staticmethod
    def get_random_subsegment(array, desired_len, tot_length):

        offset = 0
        if desired_len < tot_length:
            offset = np.random.randint(0, tot_length - desired_len)

        out, _ = sf.read(array, start=offset, stop=offset + desired_len, dtype="float32")

        if len(out.shape) > 1:
            out = out[:, random.randint(0, 1)]

        return out

    def __getitem__(self, idx):
        """Gets a mixture/sources pair.
        Returns:
            mixture, vstack([source_arrays])
        """
        if self.use_original == True:
            if random.random() <= self.orig_percentage:  # if true sample wham example
                mix_file, mixlen = self.wham_mix[idx]

                offset = 0
                if self.seg_len < mixlen:
                    offset = np.random.randint(0, mixlen - self.seg_len)

                x, _ = sf.read(mix_file, start=offset, stop=offset + self.seg_len, dtype="float32")

                seg_len = torch.as_tensor([len(x)])
                # Load sources
                source_arrays = []
                for src in self.wham_sources:
                    if src[idx] is None:
                        # Target is filled with zeros if n_src > default_nsrc
                        s = np.zeros((seg_len,))
                    else:
                        s, _ = sf.read(
                            src[idx][0], start=offset, stop=offset + self.seg_len, dtype="float32"
                        )
                    source_arrays.append(s)
                sources = torch.from_numpy(np.vstack(source_arrays))
                return torch.from_numpy(x), sources

        # else return augmented data: Sample k speakers randomly
        c_speakers = np.random.choice(
            [x for x in self.hashtab_synth.keys() if x != "noise"], self.n_src
        )

        sources = []
        first_lvl = None
        floor, ceil = self.global_db_range
        for i, spk in enumerate(c_speakers):
            tmp, tmp_spk_len = random.choice(self.hashtab_synth[c_speakers[i]])
            # account for sample reduction in speed perturb
            if self.speed_perturb:
                c_speed = random.uniform(*self.speed_perturb)
                target_len = int(np.ceil(c_speed * self.seg_len))
            else:
                target_len = self.seg_len
            tmp = self.get_random_subsegment(tmp, target_len, tmp_spk_len)
            if i == 0:  # we model the signal level distributions with gaussians
                c_lvl = np.clip(random.normalvariate(*self.abs_stats), floor, ceil)
                first_lvl = c_lvl
            else:
                c_lvl = np.clip(first_lvl - random.normalvariate(*self.rel_stats), floor, ceil)
            tmp = self.random_data_augmentation(tmp, c_lvl, c_speed)
            tmp = tmp[: self.seg_len]
            sources.append(tmp)

        if self.task in ["sep_noisy", "enh_single", "enh_both", "enhance_single"]:
            # add also noise
            tmp, tmp_spk_len = random.choice(self.hashtab_synth["noise"])
            if self.speed_perturb:
                c_speed = random.uniform(*self.speed_perturb)
                target_len = int(np.ceil(c_speed * self.seg_len))
            else:
                target_len = self.seg_len
            tmp = self.get_random_subsegment(tmp, target_len, tmp_spk_len)
            c_lvl = np.clip(first_lvl - random.normalvariate(*self.noise_stats), floor, ceil)
            tmp = self.random_data_augmentation(tmp, c_lvl, c_speed)
            tmp = tmp[: self.seg_len]
            sources.append(tmp)

        mix = np.sum(np.stack(sources), 0)

        if self.task in ["sep_noisy", "enh_single", "enhance_single", "enh_both"]:
            sources = sources[:-1]  # discard noise

        # check for clipping
        absmax = np.max(np.abs(mix))
        if absmax > 1:
            mix = mix / absmax
            sources = [x / absmax for x in sources]

        sources = np.stack(sources)

        return torch.from_numpy(mix).float(), torch.from_numpy(sources).float()
