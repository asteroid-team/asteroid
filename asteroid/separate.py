import os
import warnings
import torch
import numpy as np
import soundfile as sf
from typing import Optional

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

    in_channels: Optional[int]

    def forward_wav(self, wav: torch.Tensor, **kwargs) -> torch.Tensor:
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
    def sample_rate(self) -> float:
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
    if model.in_channels is not None and wav.shape[-2] != model.in_channels:
        raise RuntimeError(
            f"Model supports {model.in_channels}-channel inputs but found audio with {wav.shape[-2]} channels."
            f"Please match the number of channels."
        )
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

    if not hasattr(model, "sample_rate"):
        raise TypeError(
            f"This function requires your model ({type(model).__name__}) to have a "
            "'sample_rate' attribute. See `BaseModel.sample_rate` for details."
        )

    # Estimates will be saved as filename_est1.wav etc...
    base, _ = os.path.splitext(filename)
    if output_dir is not None:
        base = os.path.join(output_dir, os.path.basename(base))
    save_name_template = base + "_est{}.wav"

    # Bail out early if an estimate file already exists and we shall not overwrite.
    est1_filename = save_name_template.format(1)
    if os.path.isfile(est1_filename) and not force_overwrite:
        warnings.warn(
            f"File {est1_filename} already exists, pass `force_overwrite=True` to overwrite it",
            UserWarning,
        )
        return

    # SoundFile wav shape: [time, n_chan]
    wav, fs = _load_audio(filename)
    if wav.shape[-1] > 1:
        warnings.warn(
            f"Received multichannel signal with {wav.shape[-1]} signals, "
            f"using the first channel only."
        )
    # FIXME: support only single-channel files for now.
    if resample:
        wav = _resample(wav[:, 0], orig_sr=fs, target_sr=int(model.sample_rate))[:, None]
    elif fs != model.sample_rate:
        raise RuntimeError(
            f"Received a signal with a sampling rate of {fs}Hz for a model "
            f"of {model.sample_rate}Hz. You can pass `resample=True` to resample automatically."
        )
    # Pass wav as [batch, n_chan, time]; here: [1, chan, time]
    wav = wav.T[None]
    (est_srcs,) = numpy_separate(model, wav, **kwargs)
    # Resample to original sr
    est_srcs = [
        _resample(est_src, orig_sr=int(model.sample_rate), target_sr=fs) for est_src in est_srcs
    ]

    # Save wav files to filename_est1.wav etc...
    for src_idx, est_src in enumerate(est_srcs, 1):
        sf.write(save_name_template.format(src_idx), est_src, fs)


def _resample(wav, orig_sr, target_sr, _resamplers={}):
    from julius import ResampleFrac

    if orig_sr == target_sr:
        return wav

    # Cache ResampleFrac instance to speed up resampling if we're repeatedly
    # resampling between the same two sample rates.
    try:
        resampler = _resamplers[(orig_sr, target_sr)]
    except KeyError:
        resampler = _resamplers[(orig_sr, target_sr)] = ResampleFrac(orig_sr, target_sr)

    return resampler(torch.from_numpy(wav)).numpy()


def _load_audio(filename):
    try:
        return sf.read(filename, dtype="float32", always_2d=True)
    except Exception as sf_err:
        # If soundfile fails to load the file, try with librosa next, which uses
        # the 'audioread' library to support a wide range of audio formats.
        # We try with soundfile first because librosa takes a long time to import.
        try:
            import librosa
        except ModuleNotFoundError:
            raise RuntimeError(
                f"Could not load file {filename!r} with soundfile. "
                "Install 'librosa' to be able to load more file types."
            ) from sf_err

        wav, sr = librosa.load(filename, dtype="float32", sr=None)
        # Always return wav of shape [time, n_chan]
        if wav.ndim == 1:
            return wav[:, None], sr
        else:
            return wav.T, sr
