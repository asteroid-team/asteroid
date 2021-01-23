import sys
import torch
import numpy as np
from pathlib import Path
from catalyst.dl.core import Callback, MetricCallback, CallbackOrder

from train import snr, sdr


class SNRCallback(MetricCallback):
    """SNR callback.

    Args:
        input_key (str): input key to use for dice calculation;
            specifies our y_true.
        output_key (str): output key to use for dice calculation;
            specifies our y_pred.

    """

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "snr",
        mixed_audio_key: str = "input_audio",
    ):
        self.mixed_audio_key = mixed_audio_key
        super().__init__(prefix=prefix, metric_fn=snr, input_key=input_key, output_key=output_key)

    def on_batch_end(self, state):
        output_audios = state.output[self.output_key]
        true_audios = state.input[self.input_key]

        if hasattr(state.model, "module"):
            num_person = state.model.module.num_person
        else:
            num_person = state.model.num_person

        avg_snr = 0
        for n in range(num_person):
            output_audio = output_audios[:, n, ...]
            true_audio = true_audios[:, n, ...]

            snr_value = snr(output_audio, true_audio).item()
            avg_snr += snr_value

        avg_snr /= num_person
        state.metrics.add_batch_value(name=self.prefix, value=avg_snr)


class SDRCallback(MetricCallback):
    """SDR callback.

    Args:
        input_key (str): input key to use for dice calculation;
            specifies our y_true.
        output_key (str): output key to use for dice calculation;
            specifies our y_pred.

    """

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "sdr",
        mixed_audio_key: str = "input_audio",
    ):
        self.mixed_audio_key = mixed_audio_key
        super().__init__(prefix=prefix, metric_fn=snr, input_key=input_key, output_key=output_key)

    def on_batch_end(self, state):
        output_audios = state.output[self.output_key]
        true_audios = state.input[self.input_key]

        if hasattr(state.model, "module"):
            num_person = state.model.module.num_person
        else:
            num_person = state.model.num_person

        batch = output_audios.shape[0]
        avg_sdr = 0
        for n in range(batch):
            output_audio = output_audios[n, ...]
            true_audio = true_audios[n, ...]

            output_audio = output_audio.detach().cpu().numpy()
            true_audio = true_audio.detach().cpu().numpy()

            sdr_value = sdr(output_audio, true_audio)
            sdr_value = np.mean(sdr_value)
            avg_sdr += sdr_value

        avg_sdr /= batch
        state.metrics.add_batch_value(name=self.prefix, value=avg_sdr)
