from torch.utils.data import Dataset
import torch
import glob
import soundfile as sf
import numpy as np
import json
from scipy.signal import fftconvolve
from scipy.signal import firwin2
from pysndfx import AudioEffectsChain
from asteroid.data.librimix_dataset import librispeech_license
from asteroid.data.fuss_dataset import fuss_license

# We approximate the effect of a surgical or tissue mask with an ad-hoc FIR
# filter whose frequency response is taken from [1].
#
# [1] Corey, Ryan M., Uriah Jones, and Andrew C. Singer.
# "Acoustic effects of medical, cloth, and transparent face masks on speech signals."
#  arXiv preprint arXiv:2008.04521 (2020).

mask_firs = {
    "surgical": {
        "gain": [1, 1, 1, 1, 0.94, 0.82, 0.53, 0.80, 0.56, 0.5, 0.42, 0.63, 0.71, 0],
        "freq": [0, 250, 1000, 1500, 2000, 2500, 2815, 3175, 3500, 4000, 5000, 6000, 7000, 8000],
    },
    "KN95": {
        "gain": [1, 1, 0.95, 0.92, 0.76, 0.86, 0.36, 0.82, 0.45, 0.5, 0.47, 0.52, 0.66, 0],
        "freq": [0, 250, 1000, 1500, 2000, 2500, 2815, 3175, 3500, 4000, 5000, 6000, 7000, 8000],
    },
    "N95": {
        "gain": [1, 1, 0.95, 0.92, 0.57, 0.54, 0.3, 0.41, 0.28, 0.38, 0.44, 0.62, 0.52, 0],
        "freq": [0, 250, 1000, 1500, 2000, 2500, 2815, 3175, 3500, 4000, 5000, 6000, 7000, 8000],
    },
    "CottonJersey2L": {
        "gain": [1, 1, 1, 0.95, 0.94, 0.71, 0.61, 0.76, 0.25, 0.52, 0.46, 0.42, 0.46, 0],
        "freq": [0, 250, 1000, 1500, 2000, 2500, 2815, 3175, 3500, 4000, 5000, 6000, 7000, 8000],
    },
    "CottonPlain2L": {
        "gain": [1, 1, 1, 0.95, 0.82, 0.77, 0.47, 0.73, 0.45, 0.58, 0.46, 0.52, 0.54, 0],
        "freq": [0, 250, 1000, 1500, 2000, 2500, 2815, 3175, 3500, 4000, 5000, 6000, 7000, 8000],
    },
    "CottonSpandex3L": {
        "gain": [1, 1, 1, 1, 0.66, 0.71, 0.31, 0.68, 0.17, 0.54, 0.44, 0.39, 0.47, 0],
        "freq": [0, 250, 1000, 1500, 2000, 2500, 2815, 3175, 3500, 4000, 5000, 6000, 7000, 8000],
    },
    "CottonSpandex2L": {
        "gain": [1, 1, 1, 0.79, 0.7, 0.52, 0.37, 0.52, 0.34, 0.29, 0.34, 0.37, 0.43, 0],
        "freq": [0, 250, 1000, 1500, 2000, 2500, 2815, 3175, 3500, 4000, 5000, 6000, 7000, 8000],
    },
    "CottonDenim2L": {
        "gain": [1, 1, 1, 0.55, 0.3, 0.5, 0.11, 0.5, 0.11, 0.44, 0.22, 0.2, 0.33, 0],
        "freq": [0, 250, 1000, 1500, 2000, 2500, 2815, 3175, 3500, 4000, 5000, 6000, 7000, 8000],
    },
    "BedsheetPolyester2L": {
        "gain": [1, 1, 1, 1, 0.42, 0.15, 0.19, 0.34, 0.23, 0.23, 0.15, 0.15, 0.19, 0],
        "freq": [0, 250, 1000, 1500, 2000, 2500, 2815, 3175, 3500, 4000, 5000, 6000, 7000, 8000],
    },
}


class DeMaskDataset(Dataset):

    dataset_name = "Surgical_mask_speech_enhancement_v1"

    def __init__(
        self, configs, clean_speech_dataset, train, rirs_dataset=None, noises_dataset=None
    ):
        self.configs = configs
        self.train = train

        with open(clean_speech_dataset, "r") as f:
            clean = json.load(f)

        self.clean = []
        for c in clean:
            if c["length"] < self.configs["data"]["fs"] * self.configs["data"]["length"]:
                continue
            self.clean.append(c["file"])

        self.firs = mask_firs

        self.rirs = None
        if rirs_dataset:
            with open(rirs_dataset, "r") as f:
                rirs = json.load(f)
            self.rirs = [x["file"] for x in rirs]

        self.noises = None
        if noises_dataset:
            with open(noises_dataset, "r") as f:
                noises = json.load(f)
                for n in noises:
                    if n["length"] < self.configs["data"]["fs"] * self.configs["data"]["length"]:
                        continue
                    self.clean.append(n["file"])

    def __len__(self):
        return len(self.clean)

    def augment_clean(self, clean):

        speed = eval(self.configs["training"]["speed_augm"])
        c_gain = eval(self.configs["training"]["gain_augm"])

        fx = AudioEffectsChain().speed(speed)  # speed perturb
        clean = fx(clean)

        if self.rirs:
            c_rir = np.random.choice(self.rirs, 1)[0]
            c_rir, fs = sf.read(c_rir)
            assert fs == self.configs["data"]["fs"]
            clean = fftconvolve(clean, c_rir)

        fx = AudioEffectsChain().custom("norm {}".format(c_gain))  # random gain
        clean = fx(clean)

        return clean, c_gain

    def add_noise(self, clean, masked, c_gain):

        fx = AudioEffectsChain().custom(
            "norm {}".format(eval(c_gain - self.configs["training"]["snr"]))
        )

        noise = np.random.choice(self.noises, 1)[0]
        noise, fs = sf.read(noise)
        if len(noise.shape) > 2:
            # select random channel
            noise = noise[:, np.random.randint(0, noise.shape - 1)]
        offset = 0
        target_len = int(self.configs["data"]["fs"] * self.configs["data"]["length"])
        if len(noise) > target_len:
            offset = np.random.randint(0, len(noise) - target_len)

        noise = noise[offset : offset + target_len]
        assert fs == self.configs["data"]["fs"]
        noise = fx(noise)
        clean = clean + noise  # NB demask is not doing denoising
        masked = masked + noise

        return clean, masked

    def __getitem__(self, item):
        # 1 we sample a clean utterance
        clean = self.clean[item]
        clean, fs = sf.read(clean)
        assert fs == self.configs["data"]["fs"]
        # we sample a random window
        target_len = int(self.configs["data"]["fs"] * self.configs["data"]["length"])
        offset = 0
        if len(clean) > target_len:
            offset = np.random.randint(0, len(clean) - target_len)

        clean = clean[offset : offset + target_len]
        clean, c_gain = self.augment_clean(clean)

        # we add reverberation, speed perturb and random scaling
        masks = list(self.firs.keys())
        c_mask = np.random.choice(masks, 1)[0]
        c_mask = self.firs[c_mask]

        gains = np.array(c_mask["gain"])
        freqs = np.array(c_mask["freq"])

        if self.train:
            # augment the gains with random noise: no mask is created equal
            snr = 10 ** (eval(self.configs["training"]["gaussian_mask_noise_snr_dB"]) / 20)
            gains += np.random.normal(0, np.var(gains) / snr, gains.shape)

        fir = firwin2(
            self.configs["training"]["n_taps"], freqs, gains, fs=self.configs["data"]["fs"]
        )

        masked = fftconvolve(clean, fir)
        clean = np.pad(clean, ((len(fir) - 1) // 2, 0), mode="constant")
        trim_start = (len(fir) - 1) // 2
        trim_end = len(clean) - len(fir) + 1
        clean = clean[trim_start:trim_end]
        masked = masked[trim_start:trim_end]

        if len(clean) > target_len:
            clean = clean[:target_len]
            masked = masked[:target_len]
        elif len(clean) < target_len:
            clean = np.pad(
                clean,
                (0, int(self.configs["data"]["fs"] * self.configs["data"]["length"]) - len(clean)),
                mode="constant",
            )
            masked = np.pad(
                masked,
                (0, int(self.configs["data"]["fs"] * self.configs["data"]["length"]) - len(masked)),
                mode="constant",
            )
        else:
            pass

        if self.noises:
            clean, masked = self.add_noise(clean, masked, c_gain)
        else:  # if no noises still add gaussian noise when training
            if self.train:
                snr = 10 ** (eval(self.configs["training"]["white_noise_dB"]) / 20)
                noise = np.random.normal(0, np.var(masked) / snr, masked.shape)
                masked += noise  # NB demask is not doing denoising
                clean += noise

        return torch.from_numpy(masked).float(), torch.from_numpy(clean).float()

    def get_infos(self):
        """Get dataset infos (for publishing models).

        Returns:
            dict, dataset infos with keys `dataset`, `task` and `licences`.
        """
        infos = dict()
        infos["dataset"] = self.dataset_name
        infos["task"] = "enhancement"
        infos["licenses"] = [librispeech_license, fuss_license, demask_license]
        return infos


demask_license = dict(
    title="Acoustic effects of medical, cloth, and transparent face masks on speech signals",
    title_link="https://arxiv.org/abs/2008.04521",
    author="Corey, Ryan M and Jones, Uriah and Singer, Andrew C",
    license="CC BY 4.0",
    license_link="https://creativecommons.org/licenses/by/4.0/",
    non_commercial=False,
)
