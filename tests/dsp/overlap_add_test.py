import torch
from torch.testing import assert_allclose
import pytest

from asteroid.dsp.overlap_add import OverlapAddWrapper


@pytest.mark.parametrize("length", [10390, 89372])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("n_src", [1, 2])
@pytest.mark.parametrize("window", ["hanning"])
@pytest.mark.parametrize("window_size", [128])
@pytest.mark.parametrize("hop_size", [64])
def test_consistency_noweight(length, batch_size, n_src, window, window_size, hop_size):
    mix = torch.randn((batch_size, length)).reshape(batch_size, 1, -1)
    nnet = lambda x: x.unsqueeze(1).repeat(1, n_src, 1)
    oladd = OverlapAddWrapper(nnet, n_src, window_size, hop_size, window)
    oladded = oladd(mix)
    assert_allclose(mix.repeat(1, n_src, 1), oladded)