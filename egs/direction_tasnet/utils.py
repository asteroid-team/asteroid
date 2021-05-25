
from scipy.signal import get_window
import librosa.util as librosa_util

import logging
import numpy as np
import os
import json
import math
import scipy.io.wavfile as wf
import warnings
import gzip
import torch as th

from itertools import permutations
from torch.nn import L1Loss

EPS=1e-8


MAX_INT16 = np.iinfo(np.int16).max


def get_logger(name,
               format_str="%(asctime)s [%(pathname)s:%(lineno)s - %(levelname)s ] %(message)s",
               date_format="%Y-%m-%d %H:%M:%S",file=False):
    """
    Get python logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # file or console
    handler = logging.StreamHandler() if not file else logging.FileHandler(
        name)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt=format_str, datefmt=date_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def write_wav(fname, samps, fs=16000, normalize=True):
    """
    Write wav files in int16, support single/multi-channel
    """
    if normalize:
        samps = samps * MAX_INT16
    # scipy.io.wavfile.write could write single/multi-channel files
    # for multi-channel, accept ndarray [Nsamples, Nchannels]
    if samps.ndim != 1 and samps.shape[0] < samps.shape[1]:
        samps = np.transpose(samps)
        samps = np.squeeze(samps)
    # same as MATLAB and kaldi
    samps_int16 = samps.astype(np.int16)
    fdir = os.path.dirname(fname)
    if fdir and not os.path.exists(fdir):
        os.makedirs(fdir)
    # NOTE: librosa 0.6.0 seems could not write non-float narray
    #       so use scipy.io.wavfile instead
    wf.write(fname, fs, samps_int16)
    return samps


def load_json(fdir, name):
    """
    Load json as python object
    """
    path = os.path.join(fdir, name)
    if not os.path.exists(path):
        raise FileNotFoundError("Could not find json file: {}".format(path))
    with open(path, "r") as f:
        obj = json.load(f)
    return obj


def si_sdr(estimated_signal, reference_signals, scaling=True):
    """
    This is a scale invariant SDR. See https://arxiv.org/pdf/1811.02508.pdf
    or https://github.com/sigsep/bsseval/issues/3 for the motivation and
    explanation

    Input:
        estimated_signal and reference signals are (N,) numpy arrays

    Returns: SI-SDR as scalar
    """
    Rss = np.dot(reference_signals, reference_signals)
    this_s = reference_signals

    if scaling:
        # get the scaling factor for clean sources
        a = np.dot(this_s, estimated_signal) / Rss
    else:
        a = 1

    e_true = a * this_s
    e_res = estimated_signal - e_true

    Sss = (e_true**2).sum()
    Snn = (e_res**2).sum()

    SDR = 10 * math.log10(Sss/Snn)

    return SDR


def compute_sdr(gt, output):

    gt = center_trim(gt, output)
    per_channel_sdr = []
    sdr = si_sdr(output, gt)
    per_channel_sdr.append(sdr)
    return np.array(per_channel_sdr).mean()


def load_model(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        load_from = path
        if str(path).endswith(".gz"):
            load_from = gzip.open(path, "rb")
        klass, args, kwargs, state = th.load(load_from, 'cpu')
    print(klass)
    model = klass(*args, **kwargs)
    model.load_state_dict(state)
    return model

def center_trim(array, reference):
    """
    Trim a tensor to match with the dimension of `reference`.
    """
    if isinstance(array, np.ndarray):
        reference = reference.shape[-1]
        diff = array.shape[-1] - reference
    else:
        if hasattr(reference, "size"):
            reference = reference.size(-1)
        diff = array.size(-1) - reference
        if diff < 0:
            raise ValueError("tensor must be larger than reference")

    if diff:
        array = array[..., diff // 2:-(diff - diff // 2)]
    return array


def wsdr_loss(x, y_pred, y_true):
    # x: [B, 1, L]

    def sdr_fn(true, pred, eps=1e-8):
        num = th.sum(true * pred, dim=-1)
        den = th.norm(true, p=2, dim=-1) * th.norm(pred, p=2, dim=-1)
        return -(num / (den + EPS))

    # true and estimated noise
    z_true = x - y_true
    z_pred = x - y_pred

    a = th.sum(y_true**2, dim=-1) / (th.sum(y_true**2, dim=-1) + th.sum(z_true**2, dim=-1) + EPS)
    wSDR = a * sdr_fn(y_true, y_pred) + (1 - a) * sdr_fn(z_true, z_pred)
    return th.mean(wSDR)


def l1_loss(y_pred, y_true, return_est=False):
    n_speaker = y_pred.size()[1]
    perm = permutations(range(n_speaker))
    loss_list = []
    loss_func = L1Loss()
    indices = list(perm)
    for p in indices:
        loss_list.append(loss_func(y_pred[:, p, :], y_true))
    info = th.min(th.stack(loss_list), dim=0)
    loss = info.values
    index = info.indices
    y_pred = y_pred[:, indices[index.item()], :]
    if return_est:
       return loss, y_pred
    else:
       return loss


def window_sumsquare(window, n_frames, hop_length=200, win_length=800,
                     n_fft=800, dtype=np.float32, norm=None):
    """
    # from librosa 0.6
    Compute the sum-square envelope of a window function at a given hop length.
    This is used to estimate modulation effects induced by windowing
    observations in short-time fourier transforms.
    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`
    n_frames : int > 0
        The number of analysis frames
    hop_length : int > 0
        The number of samples to advance between frames
    win_length : [optional]
        The length of the window function.  By default, this matches `n_fft`.
    n_fft : int > 0
        The length of each analysis frame.
    dtype : np.dtype
        The data type of the output
    Returns
    -------
    wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
        The sum-squared envelope of the window function
    """
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = librosa_util.normalize(win_sq, norm=norm)**2
    win_sq = librosa_util.pad_center(win_sq, n_fft)

    # Fill the envelope
    for i in range(n_frames):
        sample = i * hop_length
        x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]
    return x

if __name__ == "__main__":
    pass