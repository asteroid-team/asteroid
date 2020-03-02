import torch
from torch import nn


def pad_x_to_y(x, y, axis=-1):
    """  Pad first argument to have same size as second argument

    Args:
        x (torch.Tensor): Tensor to be padded.
        y (torch.Tensor): Tensor to pad x to.
        axis (int): Axis to pad on.

    Returns:
        torch.Tensor, x padded to match y's shape.
    """
    if axis != -1:
        raise NotImplementedError
    inp_len = y.size(axis)
    output_len = x.size(axis)
    return nn.functional.pad(x, [0, inp_len - output_len])