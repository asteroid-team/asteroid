"""
Masker inputs and output masks
@author : Manuel Pariente, Inria-Nancy
"""

import torch


class InputFeeder:
    """ Transforms a complex input according to `input_mode`.
    Args:
        input_mode: string
    """
    def __init__(self, input_mode='reim'):
        self.input_mode = input_mode
        self.func, self.in_chan_mul = _inputs.get(input_mode)

    def feed_inputs(self, tf_rep, dim=1):
        return self.func(tf_rep, dim=dim)


class FBMasker:
    def __init__(self, masking_mode='reim'):
        self.masking_mode = masking_mode
        self.func, self.out_chan_mul = _masks.get(masking_mode)

    def apply_mask(self, tf_rep, dim=1):
        return self.func(tf_rep, dim=dim)


def mul_c(inp, other, dim=1):
    """
    Complex multiplication for tensors with real and imaginary parts
    concatenated on the `dim` axis.
    Args:
        inp: torch.Tensor. The first multiplicand with real and imaginary parts
            concatenated on the `dim` axis.
        other: torch.Tensor. The second multiplicand.
        dim: int. Dimension along which the real and imaginary parts are
            concatenated.
    Returns:
        torch.Tensor, the complex multiplication between `t1` and `t2`
    For now, it assumes that `other` has the same shape as `inp` along `dim`.
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
    """
    Takes the magnitude of a complex tensor with real and imaginary parts
    concatenated on the `dim` axis.
    Args:
        x: torch.Tensor. Re and Im are concatenated on the `dim` axis.
        dim: int. Dimension along which the Re and Im are concatenated.
    Returns:
        torch.Tensor, the magnitude of x.
    """
    return torch.stack(torch.chunk(x, 2, dim=dim), dim=-1).pow(2).sum(dim=-1)


def take_cat(x, dim=1):
    return torch.cat([take_mag(x, dim=dim), x], dim=dim)


_inputs = {
    'reim': (take_reim, 1),
    'mag': (take_mag, 1/2),
    'cat': (take_cat, 1 + 1/2)
}
_inputs['real'] = _inputs['reim']
_inputs['mod'] = _inputs['mag']
_inputs['concat'] = _inputs['cat']


def apply_real_mask(tf_rep, mask, dim=1):
    """
    Applies a real mask to a real representation.
    It corresponds to ReIm mask in [1].
    Args:
        tf_rep: torch.Tensor. The time frequency representation to apply the
            mask to.
        mask: torch.Tensor. The magnitude mask to be applied.
        dim: Kept to have the same interface with the other ones.
    Returns:
        torch.Tensor, `tf_rep` multiplied by the `mask`.
    """
    return tf_rep * mask


def apply_mag_mask(tf_rep, mask, dim=1):
    """
    Applies a magnitude mask to a complex representation.
    Args:
        tf_rep: torch.Tensor. The time frequency representation to apply the
            mask to. Re and Im are concatenated along `dim`.
        mask: torch.Tensor. The magnitude mask to be applied.
        dim: int. The frequency (or equivalent) dimension of both `tf_rep` and
            `mask`.
    Returns:
        torch.Tensor, `tf_rep` multiplied by the `mask`.
    If `tf_rep` has 2N elements along `dim`, `mask` has N elements, they are
    duplicated along `dim` to apply the same mask to both the Re and Im.
    """
    mask = torch.cat([mask, mask], dim=dim)
    return tf_rep * mask


def apply_complex_mask(tf_rep, mask, dim=1):
    """
    Applies a complex mask to a complex representation.
    Args:
        tf_rep: torch.Tensor. The time frequency representation to apply the
            mask to. Re and Im and concatenated along `dim`.
        mask: torch.Tensor. The complex mask to be applied. Re and Im are
            concatenated along `dim`.
        dim: int. The frequency (or equivalent) dimension of both `tf_rep` and
            `mask`.
    Returns:
        torch.Tensor, `tf_rep` multiplied by the `mask` in the complex sense.
    """
    return mul_c(tf_rep, mask, dim=dim)


_masks = {
    'reim': (apply_real_mask, 1),
    'mag': (apply_mag_mask, 1/2),
    'complex': (apply_complex_mask, 1)
}
_masks['real'] = _masks['reim']
_masks['mod'] = _masks['mag']
_masks['comp'] = _masks['complex']
