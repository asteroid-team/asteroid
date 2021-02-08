import torch
import torch.nn.functional as F


def xcorr(inp, ref, normalized=True, eps=1e-8):
    r"""Multi-channel cross correlation.

    The two signals can have different lengths but the input signal should be shorter than the reference signal.

    .. note:: The cross correlation is computed between each pair of microphone channels and not
        between all possible pairs e.g. if both input and ref have shape ``(1, 2, 100)``
        the output will be ``(1, 2, 1)`` the first element is the xcorr between
        the first mic channel of input and the first mic channel of ref.
        If either input and ref have only one channel e.g. input: (1, 3, 100) and ref: ``(1, 1, 100)``
        then output will be ``(1, 3, 1)`` as ref will be broadcasted to have same shape as input.

    Args:
        inp (:class:`torch.Tensor`): multi-channel input signal. Shape: :math:`(batch, mic\_channels, seq\_len)`.
        ref (:class:`torch.Tensor`): multi-channel reference signal. Shape: :math:`(batch, mic\_channels, seq\_len)`.
        normalized (bool, optional): whether to normalize the cross-correlation with the l2 norm of input signals.
        eps (float, optional): machine epsilon used for numerical stabilization when normalization is used.

    Returns:
        out (:class:`torch.Tensor`): cross correlation between the two multi-channel signals.
            Shape: :math:`(batch, mic\_channels, seq\_len\_ref - seq\_len\_input + 1)`.

    """
    # inp: batch, nmics2, seq_len2 || ref: batch, nmics1, seq_len1
    assert inp.size(0) == ref.size(0), "ref and inp signals should have same batch size."
    assert inp.size(2) >= ref.size(2), "Input signal should be shorter than the ref signal."

    inp = inp.permute(1, 0, 2).contiguous()
    ref = ref.permute(1, 0, 2).contiguous()
    bsz = inp.size(1)
    inp_mics = inp.size(0)

    if ref.size(0) > inp.size(0):
        inp = inp.expand(ref.size(0), inp.size(1), inp.size(2)).contiguous()  # nmic2, L, seg1
        inp_mics = ref.size(0)
    elif ref.size(0) < inp.size(0):
        ref = ref.expand(inp.size(0), ref.size(1), ref.size(2)).contiguous()  # nmic1, L, seg2
    # cosine similarity
    out = F.conv1d(
        inp.view(1, -1, inp.size(2)), ref.view(-1, 1, ref.size(2)), groups=inp_mics * bsz
    )  # 1, inp_mics*L, seg1-seg2+1

    # L2 norms
    if normalized:
        inp_norm = F.conv1d(
            inp.view(1, -1, inp.size(2)).pow(2),
            torch.ones(inp.size(0) * inp.size(1), 1, ref.size(2)).type(inp.type()),
            groups=inp_mics * bsz,
        )  # 1, inp_mics*L, seg1-seg2+1
        inp_norm = inp_norm.sqrt() + eps
        ref_norm = ref.norm(2, dim=2).view(1, -1, 1) + eps  # 1, inp_mics*L, 1
        out = out / (inp_norm * ref_norm)
    return out.view(inp_mics, bsz, -1).permute(1, 0, 2).contiguous()
