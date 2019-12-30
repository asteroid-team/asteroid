import sys
sys.path.append("loader/")

from audio_feature_generator import convert_to_wave

import torch
import numpy as np
from pathlib import Path


class SaveAudio:

    def __init__(self, dirname: str, filename_prefix: str, input_audio_size: int=2):
        self.dirname = Path(dirname)
        self.filename_prefix = filename_prefix
        self.input_audio_size = input_audio_size

        if not self.dirname.is_dir():
            self.dirname.mkdir()

    def __call__(self, engine):
        batch_y = engine.state.output["y"].cpu().detach().numpy()
        batch_y_pred = engine.state.output["y_pred"].cpu().detach().numpy()
        
        batch_size = batch_y.shape[0]

        y_wav = np.zeros((batch_size, self.input_audio_size, 48000, 2))

        for i, (y, y_pred) in enumerate(zip(batch_y, batch_y_pred)):
            for j in range(self.input_audio_size):
                y_temp = y[..., j].transpose(2, 1, 0)
                y_wav[i, j, ...] = np.repeat(convert_to_wave(y_temp)[:48000][..., None], 2, axis=1)

        np.save(self.dirname / f"{self.filename_prefix}{engine.state.epoch}_{engine.state.iteration}.npy", y_wav)


