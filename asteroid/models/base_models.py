import torch
from torch import nn
import numpy as np

from .. import torch_utils


class BaseTasNet(nn.Module):
    """ Base class for encoder-masker-decoder separation models.

    Args:
        encoder (Encoder): Encoder instance.
        masker (nn.Module): masker network.
        decoder (Decoder): Decoder instance.
    """
    def __init__(self, encoder, masker, decoder):
        super().__init__()
        self.encoder = encoder
        self.masker = masker
        self.decoder = decoder

    def forward(self, wav):
        """ Enc/Mask/Dec model forward

        Args:
            wav (torch.Tensor): waveform tensor. 1D, 2D or 3D tensor, time last.

        Returns:
            torch.Tensor, of shape (batch, n_src, time) or (n_src, time).
        """
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
        """ Infer separated sources from input waveforms.

        Args:
            wav (Union[torch.Tensor, numpy.ndarray]): waveform array/tensor.
                Shape: 1D, 2D or 3D tensor, time last.

        Returns:
            Union[torch.Tensor, numpy.ndarray], the estimated sources.
                (batch, n_src, time) or (n_src, time) w/o batch dim.
        """
        return self._separate(wav)

    def _separate(self, wav):
        """ Hidden separation method

        Args:
            wav (Union[torch.Tensor, numpy.ndarray]): waveform array/tensor.
                Shape: 1D, 2D or 3D tensor, time last.

        Returns:
            Union[torch.Tensor, numpy.ndarray], the estimated sources.
                (batch, n_src, time) or (n_src, time) w/o batch dim.
        """
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
    def from_pretrained(cls, pretrained_model_conf_or_path):
        """ Instantiate separation model from a model config (file or dict).

        Args:
            pretrained_model_conf_or_path (Union[dict, str]): model conf as
                returned by `serialize` or path to it.

        Returns:
            Instance of BaseTasNet
        """
        if isinstance(pretrained_model_conf_or_path, str):
            conf = torch.load(pretrained_model_conf_or_path, map_location='cpu')
        else:
            conf = pretrained_model_conf_or_path
        model = cls(**conf['fb_conf'], **conf['masker_conf'])
        model.load_state_dict(conf['state_dict'])
        return model

    def serialize(self):
        model_conf = dict()
        model_conf['fb_conf'] = self.encoder.filterbank.get_config()
        model_conf['masker_conf'] = self.masker.get_config()
        model_conf['state_dict'] = self.state_dict()
        return model_conf
