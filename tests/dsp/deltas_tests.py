import torch
import pytest

from asteroid.dsp.deltas import concat_deltas, compute_delta


@pytest.mark.parametrize("dim", [1, 2, -1, -2])
def test_delta(dim):
    phase = torch.randn(2, 257, 100)
    delta_phase = compute_delta(phase, dim=dim)
    assert phase.shape == delta_phase.shape


@pytest.mark.parametrize("dim", [1, 2, -1, -2])
@pytest.mark.parametrize("order", [1, 2])
def test_concat_deltas(dim, order):
    phase_shape = [2, 257, 100]
    phase = torch.randn(*phase_shape)
    cat_deltas = concat_deltas(phase, order=order, dim=dim)
    out_shape = list(phase_shape)
    out_shape[dim] = phase_shape[dim] * (1 + order)
    assert out_shape == list(cat_deltas.shape)
