import sys
sys.path.extend(["../loader"])
from audio_feature_generator import convert_to_wave

import torch
import numpy as np

from pathlib import Path
from metric_utils import snr
from catalyst.dl.core import Callback, MetricCallback, CallbackOrder

import logging

logging.basicConfig(filename='loss.log', filemode='a', format='%(name)s - %(levelname)s - %(message)s')
logging.warning('This will get logged to a file')

class SaveAudioCallback(Callback):
    
    def __init__(self, directory: Path=Path("/kaggle/working"), output_key="logits"):
        self.directory = directory
        self.predictions = []
        self.output_key = output_key
        
        super().__init__(CallbackOrder.External)
        
    def batch_spec_to_wave(self, batch_spectrogram, num_person, batch_size):
        wave = np.zeros((batch_size, 48000, num_person))
        for n in range(num_person):
            for b in range(batch_size):
                wave[b, :, n] = convert_to_wave(batch_spectrogram[b, ..., n].transpose(2, 1, 0))[:48000]
        return wave
        
    def on_batch_end(self, state):
        with torch.no_grad():
            self.predictions.append(state.output[self.output_key].cpu().detach().numpy())
    
    def on_epoch_start(self, state):
        self.predictions = []
    
    def on_epoch_end(self, state):
    
        num_person = state.model.num_person
        batch_size = self.predictions[0].shape[0]
        
        waves = np.zeros((len(self.predictions), batch_size, 48000, num_person)) # 3 second audio at 16khz
        
        for i, prediction in enumerate(self.predictions):
            batch_size = prediction.shape[0]
            waves[i, :batch_size, ...] = self.batch_spec_to_wave(prediction, num_person, batch_size)

        np.save(self.directory / f"{state.epoch_log}.npy", waves)

class SNRCallback(MetricCallback):
    """
    SNR callback.
    """

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "snr",
        mixed_audio_key: str="input_audio"
    ):
        """
        Args:
            input_key (str): input key to use for dice calculation;
                specifies our y_true.
            output_key (str): output key to use for dice calculation;
                specifies our y_pred.
        """
        self.mixed_audio_key = mixed_audio_key
        super().__init__(
            prefix=prefix,
            metric_fn=snr,
            input_key=input_key,
            output_key=output_key
        )
    
    def on_batch_end(self, state):
        output_audios = state.output[self.output_key]
        true_audios = state.input[self.input_key]
        
        num_person = state.model.num_person
        
        avg_snr = 0
        for n in range(num_person):
            output_audio = output_audios[..., n]
            true_audio = true_audios[..., n]
            
            logging.warning(torch.sum((output_audio-true_audio)**2))
            
            logging.warning(output_audio[0, 0, 0, 0])
            logging.warning('-'*10)
            logging.warning(true_audio[0, 0, 0, 0])

            #for i in output_audio.view(-1):
            #    if i > 20 or i<-20:
            #        print(i)
            #print('*'*1000)
            #for i in true_audio.view(-1):
            #    if i > 20 or i < -20:
            #        print(i)
            
            snr_value = snr(output_audio, true_audio).item()
            avg_snr += snr_value#(snr_value - avg_snr) / (n + 1)

        avg_snr /= num_person
        state.metrics.add_batch_value(name=self.prefix, value=avg_snr)
