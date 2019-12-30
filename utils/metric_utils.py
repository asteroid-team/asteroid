import sys
sys.path.append("../loader/")

#from loader/
from audio_feature_generator import convert_to_wave

import torch
import mir_eval
import numpy as np
from ignite.metrics.metric import Metric


def snr(y_pred, y):
    y_inter = y_pred - y

    true_power = (y ** 2).sum()
    inter_power = (y_inter ** 2).sum()

    snr = 10*torch.log10(true_power / inter_power)

    return snr


def bss_eval_sources(y_pred, y):
    n_sources = y_pred.shape[-1]

    y_pred_wav = np.zeros((n_sources, 48_000))
    y_wav = np.zeros((n_sources, 48_000))

    for i in range(n_sources):
        y_pred_wav[i] = convert_to_wave(y_pred[..., i])[:48000]
        y_wav[i] = convert_to_wave(y[..., i])[:48000]

    sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(y_wav, y_pred_wav)
    
    return sdr, sir, sar


class SNRMetric(Metric):

    def __init__(self):
        super(SNRMetric, self).__init__()

        self._snr_sum = 0
        self._total_num = 0

    def reset(self):
        self._snr_sum = 0
        self._total_num = 0

    def compute(self):
        return self._snr_sum / self._total_num

    def update(self, output):
        if isinstance(output, dict):
            y_pred = output["y_pred"]
            y = output["y"]
        else:
            y_pred, y = output

        assert y_pred.shape == y.shape

        snr_value = snr(y_pred, y)

        self._snr_sum += snr_value.item()
        self._total_num += y.shape[0]


class SDRMetric(Metric):
        
    def __init__(self):
        super(SDRMetric, self).__init__()

        self._sdr_sum = 0
        self._total_num = 0

    def reset(self):
        self._sdr_sum = 0
        self._total_num = 0

    def compute(self):
        return self._sdr_sum / self._total_num

    def update(self, output):
        if isinstance(output, dict):
            y_pred = output["y_pred"]
            y = output["y"]
        else:
            y_pred, y = output

        batch_size = y_pred.shape[0]

        sdr_val = 0
        for b in range(batch_size):
            y_pred_batch = y_pred[b]
            y_pred_batch = y_pred_batch.permute(2, 1, 0, 3)

            y_batch = y[b]
            y_batch = y_batch.permute(2, 1, 0, 3)

            sdr_val += np.mean(bss_eval_sources(y_pred_batch.cpu().detach().numpy(), y_batch.cpu().detach().numpy())[0])

        self._sdr_sum += sdr_val
        self._total_num += y.shape[0]

