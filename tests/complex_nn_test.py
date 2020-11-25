import torch
from torch.testing import assert_allclose
import pytest
import math

from asteroid import complex_nn as cnn
from asteroid.utils.test_utils import torch_version_tuple
from asteroid_filterbanks import transforms


def test_is_torch_complex():
    cnn.is_torch_complex(torch.randn(10, 10, dtype=torch.complex64))


def test_torch_complex_from_magphase():
    shape = (1, 257, 100)
    mag = torch.randn(shape).abs()
    phase = torch.remainder(torch.randn(shape), math.pi)
    out = cnn.torch_complex_from_magphase(mag, phase)
    assert_allclose(torch.abs(out), mag)
    assert_allclose(out.angle(), phase)


def test_as_torch_complex():
    shape = (1, 257, 100)
    mag = torch.randn(shape)
    phase = torch.randn(shape)
    # From mag and phase
    out = cnn.as_torch_complex((mag, phase))
    # From torch.complex
    out2 = cnn.as_torch_complex(out)
    assert_allclose(out, out2)
    # From torchaudio, ambiguous
    with pytest.warns(RuntimeWarning):
        out3 = cnn.as_torch_complex(torch.view_as_real(out))
    assert_allclose(out3, out)

    # From torchaudio, unambiguous
    _ = cnn.as_torch_complex(torch.randn(1, 5, 2))
    # From asteroid
    out4 = cnn.as_torch_complex(transforms.from_torchaudio(torch.view_as_real(out), dim=-2))
    assert_allclose(out4, out)


def test_as_torch_complex_raises():
    with pytest.raises(RuntimeError):
        cnn.as_torch_complex(torch.randn(1, 5, 3))
