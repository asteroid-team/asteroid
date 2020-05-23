import sys

import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt

import librosa
import librosa.display
from pysndfx import AudioEffectsChain

from local import convert_to_spectrogram

afc = AudioEffectsChain()

def filter_audio(y, sr=16_000, cutoff=15_000, low_cutoff=1, filter_order=5):
    sos = sg.butter(filter_order, [low_cutoff / sr / 2, cutoff / sr / 2], btype='band', analog=False, output='sos')
    filtered = sg.sosfilt(sos, y)

    return filtered

def shelf(y, sr=16_000, gain=5, frequency=500, slope=0.5, high_frequency=7_000):
    fx = afc.lowshelf(gain=gain, frequency=frequency, slope=slope).highshelf(gain=-gain, frequency=high_frequency, slope=slope)

    y = fx(y, sample_in=sr, sample_out=sr)

    return y

def plot_wav(y, y_filt):
    y_spec = convert_to_spectrogram(y)
    y_filt_spec = convert_to_spectrogram(y_filt)

    y_spec = y_spec[..., 0] + 1j*y_spec[..., 1]
    y_filt_spec = y_filt_spec[..., 0] + 1j*y_filt_spec[..., 1]

    plt.subplot(2, 1, 1)
    librosa.display.specshow(y_spec)
    plt.title("original spectrogram")

    plt.subplot(2, 1, 2)
    librosa.display.specshow(y_filt_spec)
    plt.title("filtered spectrogram")

    plt.show()

if __name__ == "__main__":
    y, sr = librosa.load("../output/heqIxWBhjSA_5007_4666_final_part1.wav", sr=16000)

    y_filt = filter_audio(y, sr)

    plot_wav(y, y_filt)

    librosa.output.write_wav("heqIxWBhjSA_5007_4666_final_part1.wav", y, sr=16000)
    librosa.output.write_wav("filtered.wav", np.array(y_filt, order='f'), sr=16000)
