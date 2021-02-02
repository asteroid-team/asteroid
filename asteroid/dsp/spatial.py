import torch
import torch.nn.functional as F


def xcorr(input, ref, normalized=True, eps=1e-8):
    """
    Efficient cross-correlation computation between multi-channel signals.
    Input tensors must be of shape: mic_channels, batch, seq_len.
    The two signals can have different seq_lens but the reference signal should not be shorter than the input signal.

    The length of output xcorr will be seq_len_ref - seq_len_input + 1.

    Note: that the cross correlation is computed between each pair of microphone
    channels and not between all possible pairs.
    E.g. if input and ref have shape (2, 1, 100) the output will be (2, 1, 1) the first element is the xcorr between
    the first mic channel of input and the first mic channel of ref.
    If either input and ref have only one channel e.g. input: (3, 1, 100) and ref: (1, 1, 100) then output will be
    (3, 1, 1) as ref will be broadcasted to have same shape as input.

    Args:
        input (:class:`torch.Tensor`): multi-channel input signal. Must be a tensor of shape (mic_channels, batch, seq_len).
        ref (:class:`torch.Tensor`): multi-channel reference signal. Must be a tensor of shape (mic_channels, batch, seq_len)
        normalized (bool, optional): whether to normalize the cross-correlation with the l2 norm of input signals.
        eps (float, optional): machine epsilon used for numerical stabilization when normalization is used.

    Returns:
        out (:class:`torch.Tensor`): cross correlation between the two multi-channel signals.

    """
    # input has size nmics2, batches, seq_len2
    # ref has size nmics1, batches, seq_len1

    assert ref.size(1) == input.size(1), "Input and reference signals should have same batch size."
    assert ref.size(2) >= input.size(
        2
    ), "Reference signal should not be shorter than the input signal."

    bsz = ref.size(1)

    ref_mics = ref.size(0)
    if input.size(0) > ref.size(0):
        ref = ref.expand(input.size(0), ref.size(1), ref.size(2)).contiguous()  # nmic2, L, seg1
        ref_mics = input.size(0)
    elif input.size(0) < ref.size(0):
        input = input.expand(
            ref.size(0), input.size(1), input.size(2)
        ).contiguous()  # nmic1, L, seg2

    # L2 norms
    if normalized:
        ref_norm = F.conv1d(
            ref.view(1, -1, ref.size(2)).pow(2),
            torch.ones(ref.size(0) * ref.size(1), 1, input.size(2)).type(ref.type()),
            groups=ref_mics * bsz,
        )  # 1, ref_mics*L, seg1-seg2+1
        ref_norm = ref_norm.sqrt() + eps
        input_norm = input.norm(2, dim=2).view(1, -1, 1) + eps  # 1, ref_mics*L, 1
    # cosine similarity
    out = F.conv1d(
        ref.view(1, -1, ref.size(2)), input.view(-1, 1, input.size(2)), groups=ref_mics * bsz
    )  # 1, ref_mics*L, seg1-seg2+1
    if normalized:
        out = out / (ref_norm * input_norm)

    return out.view(ref_mics, bsz, -1)
