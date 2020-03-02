import torch
import pytest

from asteroid import torch_utils


def test_pad():
    x = torch.randn(10, 1, 16000)
    y = torch.randn(10, 1, 16234)
    padded_x = torch_utils.pad_x_to_y(x, y)
    assert padded_x.shape == y.shape


def test_pad_fail():
    x = torch.randn(10, 16000, 1)
    y = torch.randn(10, 16234, 1)
    with pytest.raises(NotImplementedError):
        padded_x = torch_utils.pad_x_to_y(x, y, axis=1)
