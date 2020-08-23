import torch
from torch import nn
import numpy as np

from .. import torch_utils
from ..utils.hub_utils import cached_download
from ..masknn import activations


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def separate(self, wav, *args, **kwargs):
        """ Infer separated sources from input waveforms.
        Also supports filenames.

        Args:
            wav (Union[torch.Tensor, numpy.ndarray, str]): waveform array/tensor.
                Shape: 1D, 2D or 3D tensor, time last.

        Returns:
            Union[torch.Tensor, numpy.ndarray, None], the estimated sources.
                (batch, n_src, time) or (n_src, time) w/o batch dim.

        .. note::
            By default, `separate` calls `_separate` which calls `forward`.
            For models whose `forward` doesn't return waveform tensors,
            overwrite `_separate` to return waveform tensors.
        """
        # Handle filename inputs
        was_file = False
        if isinstance(wav, str):
            import soundfile as sf

            was_file = True
            filename = wav
            wav, fs = sf.read(wav, dtype="float32")
            wav = torch.from_numpy(wav)
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
        out_wavs = self._separate(wav, *args, **kwargs)
        # Back to input device (and numpy if necessary)
        out_wavs = out_wavs.to(input_device)
        if was_numpy:
            return out_wavs.cpu().data.numpy()
        if was_file:
            # Save wav files to filename_est1.wav etc...
            to_save = out_wavs.cpu().data.numpy()
            for src_idx, est_src in enumerate(to_save):
                base = ".".join(filename.split(".")[:-1])
                save_name = base + "_est{}.".format(src_idx + 1) + filename.split(".")[-1]
                sf.write(save_name, est_src, fs)
            return
        return out_wavs

    def _separate(self, wav, *args, **kwargs):
        """ Hidden separation method

        Args:
            wav (Union[torch.Tensor, numpy.ndarray, str]): waveform array/tensor.
                Shape: 1D, 2D or 3D tensor, time last.

        Returns:
            The output of self(wav, *args, **kwargs).
        """
        return self(wav, *args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_conf_or_path, *args, **kwargs):
        """ Instantiate separation model from a model config (file or dict).

        Args:
            pretrained_model_conf_or_path (Union[dict, str]): model conf as
                returned by `serialize`, or path to it. Need to contain
                `model_args` and `state_dict` keys.
            *args: Positional arguments to be passed to the model.
            **kwargs: Keyword arguments to be passed to the model.
                They overwrite the ones in the model package.

        Returns:
            nn.Module corresponding to the pretrained model conf/URL.

        Raises:
            ValueError if the input config file doesn't contain the keys
                `model_name`, `model_args` or `state_dict`.
        """
        from . import get  # Avoid circular imports

        if isinstance(pretrained_model_conf_or_path, str):
            cached_model = cached_download(pretrained_model_conf_or_path)
            conf = torch.load(cached_model, map_location="cpu")
        else:
            conf = pretrained_model_conf_or_path

        if "model_name" not in conf.keys():
            raise ValueError(
                "Expected config dictionary to have field "
                "model_name`. Found only: {}".format(conf.keys())
            )
        if "state_dict" not in conf.keys():
            raise ValueError(
                "Expected config dictionary to have field "
                "state_dict`. Found only: {}".format(conf.keys())
            )
        if "model_args" not in conf.keys():
            raise ValueError(
                "Expected config dictionary to have field "
                "model_args`. Found only: {}".format(conf.keys())
            )
        conf["model_args"].update(kwargs)  # kwargs overwrite config.
        # Attempt to find the model and instantiate it.
        try:
            model_class = get(conf["model_name"])
            model = model_class(*args, **conf["model_args"])
        except ValueError:  # Couldn't get the model, maybe custom.
            model = cls(*args, **conf["model_args"])  # Child class.
        model.load_state_dict(conf["state_dict"])
        return model

    def serialize(self):
        """ Serialize model and output dictionary.

        Returns:
            dict, serialized model with keys `model_args` and `state_dict`.
        """
        from .. import __version__ as asteroid_version  # Avoid circular imports
        import pytorch_lightning as pl  # Not used in torch.hub

        model_conf = dict(
            model_name=self.__class__.__name__,
            state_dict=self.get_state_dict(),
            model_args=self.get_model_args(),
        )
        # Additional infos
        infos = dict()
        infos["software_versions"] = dict(
            torch_version=torch.__version__,
            pytorch_lightning_version=pl.__version__,
            asteroid_version=asteroid_version,
        )
        model_conf["infos"] = infos
        return model_conf

    def get_state_dict(self):
        """ In case the state dict needs to be modified before sharing the model."""
        return self.state_dict()

    def get_model_args(self):
        raise NotImplementedError


class BaseTasNet(BaseModel):
    """ Base class for encoder-masker-decoder separation models.

    Args:
        encoder (Encoder): Encoder instance.
        masker (nn.Module): masker network.
        decoder (Decoder): Decoder instance.
    """

    def __init__(self, encoder, masker, decoder, encoder_activation=None):
        super().__init__()
        self.encoder = encoder
        self.masker = masker
        self.decoder = decoder

        self.encoder_activation = encoder_activation
        if encoder_activation:
            self.enc_activation = activations.get(encoder_activation)()
        else:
            self.enc_activation = activations.get("linear")()

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
        tf_rep = self.enc_activation(self.encoder(wav))
        est_masks = self.masker(tf_rep)
        masked_tf_rep = est_masks * tf_rep.unsqueeze(1)
        out_wavs = torch_utils.pad_x_to_y(self.decoder(masked_tf_rep), wav)
        if was_one_d:
            return out_wavs.squeeze(0)
        return out_wavs

    def get_model_args(self):
        """ Arguments needed to re-instantiate the model. """
        fb_config = self.encoder.filterbank.get_config()
        masknet_config = self.masker.get_config()
        # Assert both dict are disjoint
        if not all(k not in fb_config for k in masknet_config):
            raise AssertionError(
                "Filterbank and Mask network config share" "common keys. Merging them is not safe."
            )
        # Merge all args under model_args.
        model_args = {
            **fb_config,
            **masknet_config,
            "encoder_activation": self.encoder_activation,
        }
        return model_args
