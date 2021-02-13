"""
This script can be used to get dB lvl stats for WHAM sources and noise.
"""
import soundfile as sf
from glob import glob
import os
import matplotlib.pyplot as plt
import numpy as np

WHAM_ROOT = "/media/sam/Data/WSJ/wham_scripts/2speakers_wham/wav8k"

for mode in ["min", "max"]:
    for split in ["tr"]:

        noises = glob(os.path.join(WHAM_ROOT, mode, split, "noise", "*.wav"))
        s1 = glob(os.path.join(WHAM_ROOT, mode, split, "s1", "*.wav"))
        s2 = glob(os.path.join(WHAM_ROOT, mode, split, "s2", "*.wav"))

        # stat joint
        joint_src_stats = []

        for i in range(len(s1)):
            c_s1 = s1[i]
            c_s2 = os.path.join(WHAM_ROOT, mode, split, "s2", c_s1.split("/")[-1])
            noise = os.path.join(WHAM_ROOT, mode, split, "noise", c_s1.split("/")[-1])

            c_s1_audio, _ = sf.read(c_s1)
            c_s2_audio, _ = sf.read(c_s2)
            noise, _ = sf.read(noise)

            c_s1_lvl = 20 * np.log10(np.max(np.abs(c_s1_audio)))
            c_s2_lvl = 20 * np.log10(np.max(np.abs(c_s2_audio)))
            noises_lvl = 20 * np.log10(np.max(np.abs(noise)))

            joint_src_stats.append([c_s1_lvl, c_s2_lvl, noises_lvl])

        plt.hist2d(
            [x[0] for x in joint_src_stats],
            [x[1] for x in joint_src_stats],
            100,
            [[-50, 0], [-50, 0]],
        )
        plt.hist2d(
            [x[0] for x in joint_src_stats],
            [x[2] for x in joint_src_stats],
            100,
            [[-50, 0], [-50, 0]],
        )
