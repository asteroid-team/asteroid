import sys

import torch
import mir_eval
import numpy as np

from local import convert_to_wave

def snr(pred_signal: torch.Tensor, true_signal: torch.Tensor) -> torch.FloatTensor:
    """
        Calculate the Signal-to-Noise Ratio
        from two signals

        arguments:
            pred_signal: predicted signal spectrogram, expected shape: (N, 2, width, height)
            true_signal: original signal spectrogram, expected shape: (N, 2, width, height)
    """
    inter_signal = true_signal - pred_signal

    true_power = (true_signal ** 2).sum()
    inter_power = (inter_signal ** 2).sum()

    snr = 10*torch.log10(true_power / inter_power)

    return snr

def sdr(pred_signal: torch.Tensor, true_signal: torch.Tensor) -> torch.FloatTensor:
    """
        Calculate the Signal-to-Distortion Ratio
        from two signals

        arguments:
            pred_signal: predicted signal spectrogram, expected shape: (height, width, 2, num_person)
            true_signal: original signal spectrogram, expected shape: (height, width, 2, num_person)
    """
    n_sources = pred_signal.shape[-1]

    y_pred_wav = np.zeros((n_sources, 48_000))
    y_wav = np.zeros((n_sources, 48_000))

    for i in range(n_sources):
        y_pred_wav[i] = convert_to_wave(pred_signal[..., i])[:48000]
        y_wav[i] = convert_to_wave(true_signal[..., i])[:48000]
    sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(y_wav, y_pred_wav)

    return sdr

