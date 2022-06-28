import sys
import torch
import mir_eval
import numpy as np
from asteroid.data.avspeech_dataset import AVSpeechDataset


def snr(pred_signal: torch.Tensor, true_signal: torch.Tensor) -> torch.FloatTensor:
    """
    Calculate the Signal-to-Noise Ratio
    from two signals

    Args:
        pred_signal (torch.Tensor): predicted signal spectrogram.
        true_signal (torch.Tensor): original signal spectrogram.

    """
    inter_signal = true_signal - pred_signal

    true_power = (true_signal**2).sum()
    inter_power = (inter_signal**2).sum()

    snr = 10 * torch.log10(true_power / inter_power)

    return snr


def sdr(pred_signal: torch.Tensor, true_signal: torch.Tensor) -> torch.FloatTensor:
    """
    Calculate the Signal-to-Distortion Ratio
    from two signals

    Args:
        pred_signal (torch.Tensor): predicted signal spectrogram.
        true_signal (torch.Tensor): original signal spectrogram.

    """
    n_sources = pred_signal.shape[0]

    y_pred_wav = np.zeros((n_sources, 48_000))
    y_wav = np.zeros((n_sources, 48_000))

    for i in range(n_sources):
        y_pred_wav[i] = AVSpeechDataset.decode(pred_signal[i, ...]).numpy()
        y_wav[i] = AVSpeechDataset.decode(true_signal[i, ...]).numpy()
    sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(y_wav, y_pred_wav)

    return sdr
