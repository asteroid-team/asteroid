import os
import warnings

import numpy as np
import torch
from torch import nn

from ..masknn import activations
from ..utils.torch_utils import pad_x_to_y, script_if_tracing, jitable_shape
from ..utils.hub_utils import cached_download


@script_if_tracing
def _unsqueeze_to_3d(x):
    if x.ndim == 1:
        return x.reshape(1, 1, -1)
    elif x.ndim == 2:
        return x.unsqueeze(1)
    else:
        return x


class BaseModel(nn.Module):
    """Base class for serializable models.

    Defines saving/loading procedures as well as separation methods from
    file, torch tensors and numpy arrays.
    Need to overwrite the `foward` method, the `sample_rate` property and
    the `get_model_args` method.
    For models whose `forward` doesn't return waveform tensors,
    overwrite `_separate` to return waveform tensors.

    """

    def __init__(self, sample_rate: float = 8000.0):
        super().__init__()
        self.__sample_rate = sample_rate

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def sample_rate(self):
        """Operating sample rate of the model (float)."""
        return self.__sample_rate

    @sample_rate.setter
    def sample_rate(self, new_sample_rate: float):
        warnings.warn(
            "Other sub-components of the model might have a `sample_rate` "
            "attribute, be sure to modify them for consistency.",
            UserWarning,
        )
        self.__sample_rate = new_sample_rate

    @torch.no_grad()
    def separate(self, wav, output_dir=None, force_overwrite=False, resample=False, **kwargs):
        """Infer separated sources from input waveforms.
        Also supports filenames.

        Args:
            wav (Union[torch.Tensor, numpy.ndarray, str]): waveform array/tensor.
                Shape: 1D, 2D or 3D tensor, time last.
            output_dir (str): path to save all the wav files. If None,
                estimated sources will be saved next to the original ones.
            force_overwrite (bool): whether to overwrite existing files (when separating from file)..
            resample (bool): Whether to resample input files with wrong sample rate (when separating from file).
            **kwargs: keyword arguments to be passed to `_separate`.

        Returns:
            Union[torch.Tensor, numpy.ndarray, None], the estimated sources.
                (batch, n_src, time) or (n_src, time) w/o batch dim.

        .. note::
            By default, `separate` calls `_separate` which calls `forward`.
            For models whose `forward` doesn't return waveform tensors,
            overwrite `_separate` to return waveform tensors.
        """
        if isinstance(wav, str):
            self.file_separate(
                wav,
                output_dir=output_dir,
                force_overwrite=force_overwrite,
                resample=resample,
                **kwargs,
            )
        elif isinstance(wav, np.ndarray):
            return self.numpy_separate(wav, **kwargs)
        elif isinstance(wav, torch.Tensor):
            return self.torch_separate(wav, **kwargs)
        else:
            raise ValueError(
                f"Only support filenames, numpy arrays and torch tensors, received {type(wav)}"
            )

    def torch_separate(self, wav: torch.Tensor, **kwargs) -> torch.Tensor:
        """ Core logic of `separate`."""
        # Handle device placement
        input_device = wav.device
        model_device = next(self.parameters()).device
        wav = wav.to(model_device)
        # Forward
        out_wavs = self._separate(wav, **kwargs)

        # FIXME: for now this is the best we can do.
        out_wavs *= wav.abs().sum() / (out_wavs.abs().sum())

        # Back to input device (and numpy if necessary)
        out_wavs = out_wavs.to(input_device)
        return out_wavs

    def numpy_separate(self, wav: np.ndarray, **kwargs) -> np.ndarray:
        """ Numpy interface to `separate`."""
        wav = torch.from_numpy(wav)
        out_wav = self.torch_separate(wav, **kwargs)
        out_wav = out_wav.data.numpy()
        return out_wav

    def file_separate(
        self, filename: str, output_dir=None, force_overwrite=False, resample=False, **kwargs
    ) -> None:
        """ Filename interface to `separate`."""
        import soundfile as sf

        wav, fs = sf.read(filename, dtype="float32", always_2d=True)
        if wav.shape[-1] > 1:
            warnings.warn(
                f"Received multichannel signal with {wav.shape[-1]} signals, "
                f"using the first channel only."
            )
        # FIXME: support only single-channel files for now.
        wav = wav[:, 0]
        if fs != self.sample_rate:
            if resample:
                from librosa import resample

                wav = resample(wav, orig_sr=fs, target_sr=self.sample_rate)
            else:
                raise RuntimeError(
                    f"Received a signal with a sampling rate of {fs}Hz for a model "
                    f"of {self.sample_rate}Hz. You can pass `resample=True` to resample automatically."
                )
        to_save = self.numpy_separate(wav, **kwargs)

        # Save wav files to filename_est1.wav etc...
        for src_idx, est_src in enumerate(to_save):
            base = ".".join(filename.split(".")[:-1])
            save_name = base + "_est{}.".format(src_idx + 1) + filename.split(".")[-1]
            if os.path.isfile(save_name) and not force_overwrite:
                warnings.warn(
                    f"File {save_name} already exists, pass `force_overwrite=True` to overwrite it",
                    UserWarning,
                )
                return
            if output_dir is not None:
                save_name = os.path.join(output_dir, save_name.split("/")[-1])
            if fs != self.sample_rate:
                from librosa import resample

                est_src = resample(est_src, orig_sr=self.sample_rate, target_sr=fs)
            sf.write(save_name, est_src, fs)

    def _separate(self, wav, *args, **kwargs):
        """Hidden separation method

        Args:
            wav (Union[torch.Tensor, numpy.ndarray, str]): waveform array/tensor.
                Shape: 1D, 2D or 3D tensor, time last.

        Returns:
            The output of self(wav, *args, **kwargs).
        """
        return self(wav, *args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_conf_or_path, *args, **kwargs):
        """Instantiate separation model from a model config (file or dict).

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
        if "sample_rate" not in conf["model_args"]:
            # Try retrieving from pretrained models
            from ..utils.hub_utils import SR_HASHTABLE

            sr = None
            if isinstance(pretrained_model_conf_or_path, str):
                sr = SR_HASHTABLE.get(pretrained_model_conf_or_path, None)
            if sr is None:
                raise RuntimeError(
                    "Couldn't load pretrained model without sampling rate. You can either pass "
                    "`sample_rate` to the `from_pretrained` method or edit your model to include "
                    "the `sample_rate` key, or use `asteroid-register-sr model sample_rate` CLI."
                )
            conf["model_args"]["sample_rate"] = sr
        # Attempt to find the model and instantiate it.
        try:
            model_class = get(conf["model_name"])
        except ValueError:  # Couldn't get the model, maybe custom.
            model = cls(*args, **conf["model_args"])  # Child class.
        else:
            model = model_class(*args, **conf["model_args"])
        model.load_state_dict(conf["state_dict"])
        return model

    def serialize(self):
        """Serialize model and output dictionary.

        Returns:
            dict, serialized model with keys `model_args` and `state_dict`.
        """
        import pytorch_lightning as pl  # Not used in torch.hub

        from .. import __version__ as asteroid_version  # Avoid circular imports

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
        """Should return args to re-instantiate the class."""
        raise NotImplementedError


class BaseEncoderMaskerDecoder(BaseModel):
    """Base class for encoder-masker-decoder separation models.

    Args:
        encoder (Encoder): Encoder instance.
        masker (nn.Module): masker network.
        decoder (Decoder): Decoder instance.
        encoder_activation (Optional[str], optional): Activation to apply after encoder.
            See ``asteroid.masknn.activations`` for valid values.
    """

    def __init__(self, encoder, masker, decoder, encoder_activation=None):
        super().__init__(sample_rate=getattr(encoder, "sample_rate", None))
        self.encoder = encoder
        self.masker = masker
        self.decoder = decoder
        self.encoder_activation = encoder_activation
        self.enc_activation = activations.get(encoder_activation or "linear")()

    def forward(self, wav):
        """Enc/Mask/Dec model forward

        Args:
            wav (torch.Tensor): waveform tensor. 1D, 2D or 3D tensor, time last.

        Returns:
            torch.Tensor, of shape (batch, n_src, time) or (n_src, time).
        """
        # Remember shape to shape reconstruction, cast to Tensor for torchscript
        shape = jitable_shape(wav)
        # Reshape to (batch, n_mix, time)
        wav = _unsqueeze_to_3d(wav)

        # Real forward
        tf_rep = self.forward_encoder(wav)
        est_masks = self.forward_masker(tf_rep)
        masked_tf_rep = self.apply_masks(tf_rep, est_masks)
        decoded = self.forward_decoder(masked_tf_rep)

        reconstructed = pad_x_to_y(decoded, wav)
        return _shape_reconstructed(reconstructed, shape)

    def forward_encoder(self, wav):
        """Computes time-frequency representation of `wav`.

        Args:
            wav (torch.Tensor): waveform tensor in 3D shape, time last.

        Returns:
            torch.Tensor, of shape (batch, feat, seq).
        """
        tf_rep = self.encoder(wav)
        tf_rep = self.postprocess_encoded(tf_rep)
        return self.enc_activation(tf_rep)

    def postprocess_encoded(self, tf_rep):
        """Hook to perform transformations on the encoded, time-frequency domain
        representation (output of the encoder) before encoder activation is applied.

        Args:
            tf_rep (Tensor of shape (batch, freq, time)):
                Output of the encoder, before encoder activation is applied.

        Return:
            Transformed `tf_rep`
        """
        return tf_rep

    def forward_masker(self, tf_rep):
        """Estimates masks from time-frequency representation.

        Args:
            tf_rep (torch.Tensor): Time-frequency representation in (batch,
                feat, seq).

        Returns:
            torch.Tensor: Estimated masks
        """
        est_masks = self.masker(tf_rep)
        return self.postprocess_masks(est_masks)

    def postprocess_masks(self, masks):
        """Hook to perform transformations on the masks (output of the masker) before
        masks are applied.

        Args:
            masks (Tensor of shape (batch, n_src, freq, time)):
                Output of the masker

        Return:
            Transformed `masks`
        """
        return masks

    def apply_masks(self, tf_rep, est_masks):
        """Applies masks to time-frequency representation.

        Args:
            tf_rep (torch.Tensor): Time-frequency representation in (batch,
                feat, seq) shape.
            est_masks (torch.Tensor): Estimated masks.

        Returns:
            torch.Tensor: Masked time-frequency representations.
        """
        masked_tf_rep = est_masks * tf_rep.unsqueeze(1)
        return self.postprocess_masked(masked_tf_rep)

    def postprocess_masked(self, masked_tf_rep):
        """Hook to perform transformations on the masked time-frequency domain
        representation (result of masking in the time-frequency domain) before decoding.

        Args:
            masked_tf_rep (Tensor of shape (batch, n_src, freq, time)):
                Masked time-frequency representation, before decoding.

        Return:
            Transformed `masked_tf_rep`
        """
        return masked_tf_rep

    def forward_decoder(self, masked_tf_rep):
        """Reconstructs time-domain waveforms from masked representations.

        Args:
            masked_tf_rep (torch.Tensor): Masked time-frequency representation.

        Returns:
            torch.Tensor: Time-domain waveforms.
        """
        decoded = self.decoder(masked_tf_rep)
        return self.postprocess_decoded(decoded)

    def postprocess_decoded(self, decoded):
        """Hook to perform transformations on the decoded, time domain representation
        (output of the decoder) before original shape reconstruction.

        Args:
            decoded (Tensor of shape (batch, n_src, time)):
                Output of the decoder, before original shape reconstruction.

        Return:
            Transformed `decoded`
        """
        return decoded

    def get_model_args(self):
        """ Arguments needed to re-instantiate the model. """
        fb_config = self.encoder.filterbank.get_config()
        masknet_config = self.masker.get_config()
        # Assert both dict are disjoint
        if not all(k not in fb_config for k in masknet_config):
            raise AssertionError(
                "Filterbank and Mask network config share common keys. Merging them is not safe."
            )
        # Merge all args under model_args.
        model_args = {
            **fb_config,
            **masknet_config,
            "encoder_activation": self.encoder_activation,
        }
        return model_args


@script_if_tracing
def _shape_reconstructed(reconstructed, size):
    """Reshape `reconstructed` to have same size as `size`

    Args:
        reconstructed (torch.Tensor): Reconstructed waveform
        size (torch.Tensor): Size of desired waveform

    Returns:
        torch.Tensor: Reshaped waveform

    """
    if len(size) == 1:
        return reconstructed.squeeze(0)
    return reconstructed


# Backwards compatibility
BaseTasNet = BaseEncoderMaskerDecoder
