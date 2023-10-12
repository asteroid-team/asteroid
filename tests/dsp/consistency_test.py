import torch
from torch.testing import assert_close
import pytest

from asteroid.dsp.consistency import mixture_consistency


@pytest.mark.parametrize("mix_shape", [[2, 1600], [2, 130, 10]])
@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize("n_src", [1, 2, 3])
def test_consistency_noweight(mix_shape, dim, n_src):
    mix = torch.randn(mix_shape)
    est_shape = mix_shape[:dim] + [n_src] + mix_shape[dim:]
    est_sources = torch.randn(est_shape)
    consistent_est_sources = mixture_consistency(mix, est_sources, dim=dim)
    assert_close(mix, consistent_est_sources.sum(dim))


@pytest.mark.parametrize("mix_shape", [[2, 1600], [2, 130, 10]])
@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize("n_src", [1, 2, 3])
def test_consistency_withweight(mix_shape, dim, n_src):
    mix = torch.randn(mix_shape)
    est_shape = mix_shape[:dim] + [n_src] + mix_shape[dim:]
    est_sources = torch.randn(est_shape)
    # Create source weights : should have the same number of dims as
    # est_sources with ones out of batch and n_src dims.
    ones = [1 for _ in range(len(mix_shape) - 1)]
    src_weights_shape = mix_shape[:1] + ones[: dim - 1] + [n_src] + ones[dim - 1 :]
    src_weights = torch.softmax(torch.randn(src_weights_shape), dim=dim)
    # Apply mixture consitency
    consistent_est_sources = mixture_consistency(mix, est_sources, src_weights=src_weights, dim=dim)
    assert_close(mix, consistent_est_sources.sum(dim))


def test_consistency_raise():
    mix = torch.randn(10, 1, 1, 160)
    est = torch.randn(10, 2, 160)
    with pytest.raises(RuntimeError):
        mixture_consistency(mix, est, dim=1)
