import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from torch_stoi import NegSTOILoss as _NegSTOILoss


class NegSTOILoss(_NegSTOILoss):
    r"""Negated Short Term Objective Intelligibility (STOI) metric, to be used
    as a loss function.
    Inspired from [1, 2, 3] but not the same.

    Args:
        sample_rate (int): sample rate of the audio files
        use_vad (bool): Whether to use simple VAD (see Notes)
        extended (bool): Whether to compute extended version [3].

    Shapes:
        - :math:`(time,) -> (1, )`
        - :math:`(batch, time) -> (batch, )`
        - :math:`(batch, n\_src, time) -> (batch, n\_src)`

    Returns:
        torch.Tensor of shape (batch, *, ), only the time dimension has
        been reduced.

    .. warnings::
        This function cannot be used to compute the "real" STOI metric as
        we applied some changes to speed-up loss computation. See Notes section.

    .. note::
        In the NumPy version, some kind of simple VAD was used to remove the
        silent frames before chunking the signal into short-term envelope
        vectors. We don't do the same here because removing frames in a
        batch is cumbersome and inefficient.
        If `use_vad` is set to True, instead we detect the silent frames and
        keep a mask tensor. At the end, the normalized correlation of
        short-term envelope vectors is masked using this mask (unfolded) and
        the mean is computed taking the mask values into account.

    Examples
        >>> import torch
        >>> from asteroid.losses import PITLossWrapper
        >>> targets = torch.randn(10, 2, 32000)
        >>> est_targets = torch.randn(10, 2, 32000)
        >>> loss_func = PITLossWrapper(NegSTOILoss(sample_rate=8000), pit_from='pw_pt')
        >>> loss = loss_func(est_targets, targets)

    References
        [1] C.H.Taal, R.C.Hendriks, R.Heusdens, J.Jensen 'A Short-Time
        Objective Intelligibility Measure for Time-Frequency Weighted Noisy
        Speech', ICASSP 2010, Texas, Dallas.

        [2] C.H.Taal, R.C.Hendriks, R.Heusdens, J.Jensen 'An Algorithm for
        Intelligibility Prediction of Time-Frequency Weighted Noisy Speech',
        IEEE Transactions on Audio, Speech, and Language Processing, 2011.

        [3] Jesper Jensen and Cees H. Taal, 'An Algorithm for Predicting the
        Intelligibility of Speech Masked by Modulated Noise Maskers',
        IEEE Transactions on Audio, Speech and Language Processing, 2016.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
