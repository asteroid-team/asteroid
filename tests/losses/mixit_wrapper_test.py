import pytest
import itertools
import torch
from torch.testing import assert_allclose

from asteroid.losses import MixITLossWrapper, pairwise_mse


def good_batch_loss_func(y_pred, y_true):
    batch, *_ = y_true.shape
    return torch.randn(batch)


@pytest.mark.parametrize("batch_size", [1, 2, 8])
@pytest.mark.parametrize("factor", [1, 2, 3])
@pytest.mark.parametrize("n_mix", [2, 3])
@pytest.mark.parametrize("time", [16000])
def test_mixit_wrapper(batch_size, n_mix, time, factor):
    mixtures = torch.randn(batch_size, n_mix, time)
    n_src = n_mix * factor
    targets = torch.randn(batch_size, n_src, time)
    est_targets = torch.randn(batch_size, n_src, time)

    # mix_it base case: targets == mixtures / With and without return estimates
    loss = MixITLossWrapper(good_batch_loss_func, pit_from="mix_it")
    loss(est_targets, targets)
    loss_value, reordered_est = loss(est_targets, targets, return_est=True)
    assert reordered_est.shape == est_targets.shape

    # mix_it / With and without return estimates
    loss = MixITLossWrapper(good_batch_loss_func, pit_from="mix_it")
    loss(est_targets, mixtures)
    loss_value, reordered_est = loss(est_targets, mixtures, return_est=True)
    assert reordered_est.shape == mixtures.shape


@pytest.mark.parametrize("batch_size", [1, 2, 8])
@pytest.mark.parametrize("n_src", [2, 3, 4])
@pytest.mark.parametrize("n_mix", [2])
@pytest.mark.parametrize("time", [16000])
def test_mixit_gen_wrapper(batch_size, n_src, n_mix, time):
    mixtures = torch.randn(batch_size, n_mix, time)
    est_targets = torch.randn(batch_size, n_src, time)

    # mix_it_gen / With and without return estimates. Works only with two mixtures
    loss = MixITLossWrapper(good_batch_loss_func, pit_from="mix_it_gen")
    loss(est_targets, mixtures)
    loss_value, reordered_est = loss(est_targets, mixtures, return_est=True)
    assert reordered_est.shape == mixtures.shape
