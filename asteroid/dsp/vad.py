import torch
from ..utils.torch_utils import script_if_tracing


@script_if_tracing
def ebased_vad(mag_spec, th_db: int = 40):
    """Compute energy-based VAD from a magnitude spectrogram (or equivalent).

    Args:
        mag_spec (torch.Tensor): the spectrogram to perform VAD on.
            Expected shape (batch, *, freq, time).
            The VAD mask will be computed independently for all the leading
            dimensions until the last two. Independent of the ordering of the
            last two dimensions.
        th_db (int): The threshold in dB from which a TF-bin is considered
            silent.

    Returns:
        :class:`torch.BoolTensor`, the VAD mask.


    Examples
        >>> import torch
        >>> mag_spec = torch.abs(torch.randn(10, 2, 65, 16))
        >>> batch_src_mask = ebased_vad(mag_spec)
    """
    log_mag = 20 * torch.log10(mag_spec)
    # Compute VAD for each utterance in a batch independently.
    to_view = list(mag_spec.shape[:-2]) + [1, -1]
    max_log_mag = torch.max(log_mag.view(to_view), -1, keepdim=True)[0]
    return log_mag > (max_log_mag - th_db)
