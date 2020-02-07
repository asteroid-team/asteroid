import torch
EPS = 1e-8


def mul_c(inp, other, dim=1):
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
        dim (int, optional): Dimension along which the real and imaginary parts
            are concatenated.
    Returns:
        :class:`torch.Tensor`:
            The complex multiplication between `inp` and `other`

            For now, it assumes that `other` has the same shape as `inp` along
            `dim`.
    """
    if inp.shape[dim] % 2 != 0:
        raise ValueError('Shape along dimension {} should be even, received '
                         '{}'.format(dim, inp.shape[dim]))
    real1, imag1 = inp.chunk(2, dim=dim)
    real2, imag2 = other.chunk(2, dim=dim)
    return torch.cat([real1 * real2 - imag1 * imag2,
                      real1 * imag2 + imag1 * real2], dim=dim)


def take_reim(x, dim=1):
    return x


def take_mag(x, dim=1):
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
        dim (int): Dimension along which the Re and Im are concatenated.

    Returns:
        :class:`torch.Tensor`: The magnitude of x.
    """
    power = torch.stack(torch.chunk(x, 2, dim=dim), dim=-1).pow(2).sum(dim=-1)
    power = power + EPS
    return power.pow(0.5)


def take_cat(x, dim=1):
    return torch.cat([take_mag(x, dim=dim), x], dim=dim)


def apply_real_mask(tf_rep, mask, dim=1):
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


def apply_mag_mask(tf_rep, mask, dim=1):
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
            `mask`.
    Returns:
        :class:`torch.Tensor`: `tf_rep` multiplied by the `mask`.
    """
    mask = torch.cat([mask, mask], dim=dim)
    return tf_rep * mask


def apply_complex_mask(tf_rep, mask, dim=1):
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
            `mask`.

    Returns:
        :class:`torch.Tensor`:
            `tf_rep` multiplied by the `mask` in the complex sense.
    """
    return mul_c(tf_rep, mask, dim=dim)


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
