import torch
from torch.testing import assert_close
import pytest
import math

from asteroid import complex_nn as cnn
from asteroid.utils.deprecation_utils import VisibleDeprecationWarning
from asteroid_filterbanks import transforms


def test_is_torch_complex():
    cnn.is_torch_complex(torch.randn(10, 10, dtype=torch.complex64))


def test_torch_complex_from_magphase():
    shape = (1, 257, 100)
    mag = torch.randn(shape).abs()
    phase = torch.remainder(torch.randn(shape), math.pi)
    out = cnn.torch_complex_from_magphase(mag, phase)
    assert_close(torch.abs(out), mag)
    assert_close(out.angle(), phase)


def test_torch_complex_from_reim():
    comp = torch.randn(10, 12, dtype=torch.complex64)
    assert_close(cnn.torch_complex_from_reim(comp.real, comp.imag), comp)


def test_onreim():
    inp = torch.randn(10, 10, dtype=torch.complex64)
    # Identity
    fn = cnn.on_reim(lambda x: x)
    assert_close(fn(inp), inp)
    # Top right quadrant
    fn = cnn.on_reim(lambda x: x.abs())
    assert_close(fn(inp), cnn.torch_complex_from_reim(inp.real.abs(), inp.imag.abs()))


def test_on_reim_class():
    inp = torch.randn(10, 10, dtype=torch.complex64)

    class Identity(torch.nn.Module):
        def __init__(self, a=0, *args, **kwargs):
            super().__init__()
            self.a = a

        def forward(self, x):
            return x + self.a

    fn = cnn.OnReIm(Identity, 0)
    assert_close(fn(inp), inp)
    fn = cnn.OnReIm(Identity, 1)
    assert_close(fn(inp), cnn.torch_complex_from_reim(inp.real + 1, inp.imag + 1))


def test_complex_mul_wrapper():
    a = torch.randn(10, 10, dtype=torch.complex64)

    fn = cnn.ComplexMultiplicationWrapper(torch.nn.ReLU)
    assert_close(
        fn(a),
        cnn.torch_complex_from_reim(
            torch.relu(a.real) - torch.relu(a.imag), torch.relu(a.real) + torch.relu(a.imag)
        ),
    )


@pytest.mark.parametrize("bound_type", ("BDSS", "sigmoid", "BDT", "tanh", "UBD", None))
def test_bound_complex_mask(bound_type):
    cnn.bound_complex_mask(torch.randn(4, 2, 257, dtype=torch.complex64), bound_type=bound_type)


def test_bound_complex_mask_raises():
    with pytest.raises(ValueError):
        cnn.bound_complex_mask(torch.randn(4, 2, 257, dtype=torch.complex64), bound_type="foo")


@pytest.mark.parametrize("n_layers", [1, 2, 3])
def test_complexsinglernn(n_layers):
    crnn = cnn.ComplexSingleRNN("RNN", 10, 10, n_layers=n_layers, dropout=0, bidirectional=False)
    inp = torch.randn(1, 5, 10, dtype=torch.complex64)
    out = crnn(inp)
    for layer in crnn.rnns:
        rere = layer.re_module(inp.real)
        imim = layer.im_module(inp.imag)
        reim = layer.re_module(inp.imag)
        imre = layer.im_module(inp.real)
        inp = cnn.torch_complex_from_reim(rere - imim, reim + imre)
    assert_close(out, inp)
