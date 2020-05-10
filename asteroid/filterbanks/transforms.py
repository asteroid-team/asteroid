import torch
import numpy as np
EPS = 1e-8


def mul_c(inp, other, dim=-2):
    """ Entrywise product for complex valued tensors.

    Operands are assumed to have the real parts of each entry followed by the
    imaginary parts of each entry along dimension `dim`, e.g. for,
    ``dim = 1``, the matrix

    .. code::

        [[1, 2, 3, 4],
         [5, 6, 7, 8]]

    is interpreted as

    .. code::

        [[1 + 3j, 2 + 4j],
         [5 + 7j, 6 + 8j]

    where `j` is such that `j * j = -1`.

    Args:
        inp (:class:`torch.Tensor`): The first operand with real and
            imaginary parts concatenated on the `dim` axis.
        other (:class:`torch.Tensor`): The second operand.
        dim (int, optional): frequency (or equivalent) dimension along which
            real and imaginary values are concatenated.
    Returns:
        :class:`torch.Tensor`:
            The complex multiplication between `inp` and `other`

            For now, it assumes that `other` has the same shape as `inp` along
            `dim`.
    """
    check_complex(inp, dim=dim)
    check_complex(other, dim=dim)
    real1, imag1 = inp.chunk(2, dim=dim)
    real2, imag2 = other.chunk(2, dim=dim)
    return torch.cat([real1 * real2 - imag1 * imag2,
                      real1 * imag2 + imag1 * real2], dim=dim)


def take_reim(x, dim=-2):
    return x


def take_mag(x, dim=-2):
    """ Takes the magnitude of a complex tensor.

    The operands is assumed to have the real parts of each entry followed by
    the imaginary parts of each entry along dimension `dim`, e.g. for,
    ``dim = 1``, the matrix

    .. code::

        [[1, 2, 3, 4],
         [5, 6, 7, 8]]

    is interpreted as

    .. code::

        [[1 + 3j, 2 + 4j],
         [5 + 7j, 6 + 8j]

    where `j` is such that `j * j = -1`.

    Args:
        x (:class:`torch.Tensor`): Complex valued tensor.
        dim (int): frequency (or equivalent) dimension along which real and
            imaginary values are concatenated.

    Returns:
        :class:`torch.Tensor`: The magnitude of x.
    """
    check_complex(x, dim=dim)
    power = torch.stack(torch.chunk(x, 2, dim=dim), dim=-1).pow(2).sum(dim=-1)
    power = power + EPS
    return power.pow(0.5)


def take_cat(x, dim=-2):
    return torch.cat([take_mag(x, dim=dim), x], dim=dim)


def apply_real_mask(tf_rep, mask, dim=-2):
    """ Applies a real-valued mask to a real-valued representation.

    It corresponds to ReIm mask in [1].

    Args:
        tf_rep (:class:`torch.Tensor`): The time frequency representation to
            apply the mask to.
        mask (:class:`torch.Tensor`): The real-valued mask to be applied.
        dim (int): Kept to have the same interface with the other ones.
    Returns:
        :class:`torch.Tensor`: `tf_rep` multiplied by the `mask`.
    """
    return tf_rep * mask


def apply_mag_mask(tf_rep, mask, dim=-2):
    """ Applies a real-valued mask to a complex-valued representation.

    If `tf_rep` has 2N elements along `dim`, `mask` has N elements, `mask` is
    duplicated along `dim` to apply the same mask to both the Re and Im.

    `tf_rep` is assumed to have the real parts of each entry followed by
    the imaginary parts of each entry along dimension `dim`, e.g. for,
    ``dim = 1``, the matrix

    .. code::

        [[1, 2, 3, 4],
         [5, 6, 7, 8]]

    is interpreted as

    .. code::

        [[1 + 3j, 2 + 4j],
         [5 + 7j, 6 + 8j]

    where `j` is such that `j * j = -1`.

    Args:
        tf_rep (:class:`torch.Tensor`): The time frequency representation to
            apply the mask to. Re and Im are concatenated along `dim`.
        mask (:class:`torch.Tensor`): The real-valued mask to be applied.
        dim (int): The frequency (or equivalent) dimension of both `tf_rep` and
            `mask` along which real and imaginary values are concatenated.
    Returns:
        :class:`torch.Tensor`: `tf_rep` multiplied by the `mask`.
    """
    check_complex(tf_rep, dim=dim)
    mask = torch.cat([mask, mask], dim=dim)
    return tf_rep * mask


def apply_complex_mask(tf_rep, mask, dim=-2):
    """ Applies a complex-valued mask to a complex-valued representation.

    Operands are assumed to have the real parts of each entry followed by the
    imaginary parts of each entry along dimension `dim`, e.g. for,
    ``dim = 1``, the matrix

    .. code::

        [[1, 2, 3, 4],
         [5, 6, 7, 8]]

    is interpreted as

    .. code::

        [[1 + 3j, 2 + 4j],
         [5 + 7j, 6 + 8j]

    where `j` is such that `j * j = -1`.

    Args:
        tf_rep (:class:`torch.Tensor`): The time frequency representation to
            apply the mask to.
        mask (class:`torch.Tensor`): The complex-valued mask to be applied.
        dim (int): The frequency (or equivalent) dimension of both `tf_rep` an
            `mask` along which real and imaginary values are concatenated.

    Returns:
        :class:`torch.Tensor`:
            `tf_rep` multiplied by the `mask` in the complex sense.
    """
    check_complex(tf_rep, dim=dim)
    return mul_c(tf_rep, mask, dim=dim)


def check_complex(tensor, dim=-2):
    """ Assert tensor in complex-like in a given dimension.

    Args:
        tensor (torch.Tensor): tensor to be checked.
        dim(int): the frequency (or equivalent) dimension along which
            real and imaginary values are concatenated.

    Raises:
        AssertionError if dimension is not even in the specified dimension

    """
    if tensor.shape[dim] % 2 != 0:
        raise AssertionError('Could not equally chunk the tensor (shape {}) '
                             'along the given dimension ({}). Dim axis is '
                             'probably wrong')


def to_numpy(tensor, dim=-2):
    """ Convert complex-like torch tensor to numpy complex array

    Args:
        tensor (torch.Tensor): Complex tensor to convert to numpy.
        dim(int, optional): the frequency (or equivalent) dimension along which
            real and imaginary values are concatenated.
    Returns:
        :class:`numpy.array`:
            Corresponding complex array.
    """
    check_complex(tensor, dim=dim)
    real, imag = torch.chunk(tensor, 2, dim=dim)
    return real.data.numpy() + 1j * imag.data.numpy()


def from_numpy(array, dim=-2):
    """ Convert complex numpy array to complex-like torch tensor.

    Args:
        array (np.array): array to be converted.
        dim(int, optional): the frequency (or equivalent) dimension along which
            real and imaginary values are concatenated.

    Returns:
        :class:`torch.Tensor`:
            Corresponding torch.Tensor (complex axis in dim `dim`=
    """
    return torch.cat([torch.from_numpy(np.real(array)),
                      torch.from_numpy(np.imag(array))], dim=dim)


def to_torchaudio(tensor, dim=-2):
    """ Converts complex-like torch tensor to torchaudio style complex tensor.

    Args:
        tensor (torch.tensor): asteroid-style complex-like torch tensor.
        dim(int, optional): the frequency (or equivalent) dimension along which
            real and imaginary values are concatenated.

    Returns:
        :class:`torch.Tensor`:
            torchaudio-style complex-like torch tensor.
    """
    return torch.stack(torch.chunk(tensor, 2, dim=dim), dim=-1)


def from_torchaudio(tensor, dim=-2):
    """ Converts torchaudio style complex tensor to complex-like torch tensor.

    Args:
        tensor (torch.tensor): torchaudio-style complex-like torch tensor.
        dim(int, optional): the frequency (or equivalent) dimension along which
            real and imaginary values are concatenated.

    Returns:
        :class:`torch.Tensor`:
            asteroid-style complex-like torch tensor.
    """
    return torch.cat([tensor[..., 0], tensor[..., 1]], dim=dim)


def angle(tensor, dim=-2):
    """ Return the angle of the complex-like torch tensor.

    Args:
        tensor (torch.Tensor): the complex tensor from which to extract the
            phase.
        dim(int, optional): the frequency (or equivalent) dimension along which
            real and imaginary values are concatenated.

    Returns:
        :class:`torch.Tensor`:
            The counterclockwise angle from the positive real axis on
            the complex plane in radians.
    """
    check_complex(tensor, dim=dim)
    real, imag = torch.chunk(tensor, 2, dim=dim)
    return torch.atan2(imag, real)


def from_mag_and_phase(mag, phase, dim=-2):
    """ Return a complex-like torch tensor from magnitude and phase components.

    Args:
        mag (torch.tensor): magnitude of the tensor.
        phase (torch.tensor): angle of the tensor
        dim(int, optional): the frequency (or equivalent) dimension along which
            real and imaginary values are concatenated.

    Returns:
        :class:`torch.Tensor`:
            The corresponding complex-like torch tensor.
    """
    return torch.cat([mag*torch.cos(phase), mag*torch.sin(phase)], dim=dim)


def ebased_vad(mag_spec, th_db=40):
    """ Compute energy-based VAD from a magnitude spectrogram (or equivalent).

    Args:
        mag_spec (torch.Tensor): the spectrogram to perform VAD on.
            Expected shape (batch, *, freq, time).
            The VAD mask will be computed independently for all the leading
            dimensions until the last two. Independent of the ordering of the
            last two dimensions.
        th_db (int): The threshold in dB from which a TF-bin is considered
            silent.

    Returns:
        torch.BoolTensor, the VAD mask.


    Examples:
        >>> import torch
        >>> mag_spec = torch.abs(torch.randn(10, 2, 65, 16))
        >>> batch_src_mask = ebased_vad(mag_spec)
    """
    log_mag = 20 * torch.log10(mag_spec)
    # Compute VAD for each utterance in a batch independently.
    to_view = list(mag_spec.shape[:-2]) + [1, -1]
    max_log_mag = torch.max(log_mag.view(to_view), -1, keepdim=True)[0]
    return log_mag > (max_log_mag - th_db)


_inputs = {
    'reim': (take_reim, 1),
    'mag': (take_mag, 1/2),
    'cat': (take_cat, 1 + 1/2)
}
_inputs['real'] = _inputs['reim']
_inputs['mod'] = _inputs['mag']
_inputs['concat'] = _inputs['cat']


_masks = {
    'reim': (apply_real_mask, 1),
    'mag': (apply_mag_mask, 1/2),
    'complex': (apply_complex_mask, 1)
}
_masks['real'] = _masks['reim']
_masks['mod'] = _masks['mag']
_masks['comp'] = _masks['complex']
