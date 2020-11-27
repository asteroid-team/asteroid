import os
import torch
import numpy as np
import warnings

try:
    from typing import Protocol
except ImportError:  # noqa
    # Python < 3.8
    class Protocol:
        pass


from .dsp.overlap_add import LambdaOverlapAdd
from .utils import get_device


class Separatable(Protocol):
    """Things that are separatable."""

    def forward_wav(self, wav, **kwargs):
        """
        Args:
            wav (torch.Tensor): waveform tensor.
                Shape: 1D, 2D or 3D tensor, time last.
            **kwargs: Keyword arguments from `separate`.

        Returns:
            torch.Tensor: the estimated sources.
                Shape: [batch, n_src, time] or [n_src, time] if the input `wav`
                did not have a batch dim.
        """
        ...

    @property
    def sample_rate(self):
        """Operating sample rate of the model (float)."""
        ...


def separate(
    model: Separatable, wav, output_dir=None, force_overwrite=False, resample=False, **kwargs
):
    """Infer separated sources from input waveforms.
    Also supports filenames.

    Args:
        model (Separatable, for example asteroid.models.BaseModel): Model to use.
        wav (Union[torch.Tensor, numpy.ndarray, str]): waveform array/tensor.
            Shape: 1D, 2D or 3D tensor, time last.
        output_dir (str): path to save all the wav files. If None,
            estimated sources will be saved next to the original ones.
        force_overwrite (bool): whether to overwrite existing files
            (when separating from file).
        resample (bool): Whether to resample input files with wrong sample rate
            (when separating from file).
        **kwargs: keyword arguments to be passed to `forward_wav`.

    Returns:
        Union[torch.Tensor, numpy.ndarray, None], the estimated sources.
            (batch, n_src, time) or (n_src, time) w/o batch dim.

    .. note::
        `separate` calls `model.forward_wav` which calls `forward` by default.
        For models whose `forward` doesn't have waveform tensors as input/ouput,
        overwrite their `forward_wav` method to separate from waveform to waveform.
    """
    if isinstance(wav, str):
        file_separate(
            model,
            wav,
            output_dir=output_dir,
            force_overwrite=force_overwrite,
            resample=resample,
            **kwargs,
        )
    elif isinstance(wav, np.ndarray):
        return numpy_separate(model, wav, **kwargs)
    elif isinstance(wav, torch.Tensor):
        return torch_separate(model, wav, **kwargs)
    else:
        raise ValueError(
            f"Only support filenames, numpy arrays and torch tensors, received {type(wav)}"
        )


@torch.no_grad()
def torch_separate(model: Separatable, wav: torch.Tensor, **kwargs) -> torch.Tensor:
    """Core logic of `separate`."""
    # Handle device placement
    input_device = get_device(wav, default="cpu")
    model_device = get_device(model, default="cpu")
    wav = wav.to(model_device)
    # Forward
    separate_func = getattr(model, "forward_wav", model)
    out_wavs = separate_func(wav, **kwargs)

    # FIXME: for now this is the best we can do.
    out_wavs *= wav.abs().sum() / (out_wavs.abs().sum())

    # Back to input device (and numpy if necessary)
    out_wavs = out_wavs.to(input_device)
    return out_wavs


def numpy_separate(model: Separatable, wav: np.ndarray, **kwargs) -> np.ndarray:
    """Numpy interface to `separate`."""
    wav = torch.from_numpy(wav)
    out_wavs = torch_separate(model, wav, **kwargs)
    out_wavs = out_wavs.data.numpy()
    return out_wavs


def file_separate(
    model: Separatable,
    filename: str,
    output_dir=None,
    force_overwrite=False,
    resample=False,
    **kwargs,
) -> None:
    """Filename interface to `separate`."""
    import soundfile as sf

    if not hasattr(model, "sample_rate"):
        raise TypeError(
            f"This function requires your model ({type(model).__name__}) to have a "
            "'sample_rate' attribute. See `BaseModel.sample_rate` for details."
        )

    # SoundFile wav shape: [time, n_chan]
    wav, fs = sf.read(filename, dtype="float32", always_2d=True)
    if wav.shape[-1] > 1:
        warnings.warn(
            f"Received multichannel signal with {wav.shape[-1]} signals, "
            f"using the first channel only."
        )
    # FIXME: support only single-channel files for now.
    if fs != model.sample_rate:
        if resample:
            from librosa import resample

            wav = resample(wav[:, 0], orig_sr=fs, target_sr=model.sample_rate)[:, None]
        else:
            raise RuntimeError(
                f"Received a signal with a sampling rate of {fs}Hz for a model "
                f"of {model.sample_rate}Hz. You can pass `resample=True` to resample automatically."
            )
    # Pass wav as [batch, n_chan, time]; here: [1, 1, time]
    wav = wav[:, 0][None, None]
    (to_save,) = numpy_separate(model, wav, **kwargs)

    # Save wav files to filename_est1.wav etc...
    for src_idx, est_src in enumerate(to_save):
        base = ".".join(filename.split(".")[:-1])
        save_name = base + "_est{}.".format(src_idx + 1) + filename.split(".")[-1]
        if output_dir is not None:
            save_name = os.path.join(output_dir, save_name.split("/")[-1])
        if os.path.isfile(save_name) and not force_overwrite:
            warnings.warn(
                f"File {save_name} already exists, pass `force_overwrite=True` to overwrite it",
                UserWarning,
            )
            return
        if fs != model.sample_rate:
            from librosa import resample

            est_src = resample(est_src, orig_sr=model.sample_rate, target_sr=fs)
        sf.write(save_name, est_src, fs)
