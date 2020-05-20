import torch
from torch import nn
import numpy as np

from .. import torch_utils


class BaseTasNet(nn.Module):
    def __init__(self, encoder, masker, decoder):
        super().__init__()
        self.encoder = encoder
        self.masker = masker
        self.decoder = decoder

    def forward(self, wav):
        # Handle 1D, 2D or n-D inputs
        was_one_d = False
        if wav.ndim == 1:
            was_one_d = True
            wav = wav.unsqueeze(0).unsqueeze(1)
        if wav.ndim == 2:
            wav = wav.unsqueeze(1)
        # Real forward
        tf_rep = self.encoder(wav)
        est_masks = self.masker(tf_rep)
        masked_tf_rep = est_masks * tf_rep.unsqueeze(1)
        out_wavs = torch_utils.pad_x_to_y(self.decoder(masked_tf_rep), wav)
        if was_one_d:
            return out_wavs.squeeze(0)
        return out_wavs

    def separate(self, wav):
        # Handle numpy inputs
        was_numpy = False
        if isinstance(wav, np.ndarray):
            was_numpy = True
            wav = torch.from_numpy(wav)
        # Handle device placement
        input_device = wav.device
        model_device = next(self.parameters()).device
        wav = wav.to(model_device)
        # Forward
        out_wavs = self.forward(wav)
        # Back to input device (and numpy if necessary)
        out_wavs = out_wavs.to(input_device)
        if was_numpy:
            return out_wavs.cpu().data.numpy()
        return out_wavs

    @classmethod
    def from_pretrained(cls):
        raise NotImplementedError
