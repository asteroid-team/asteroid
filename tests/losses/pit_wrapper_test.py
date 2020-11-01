import pytest
import itertools
import torch
from torch.testing import assert_allclose

from asteroid.losses import PITLossWrapper, pairwise_mse


def bad_loss_func_ndim0(y_pred, y_true):
    return torch.randn(1).mean()


def bad_loss_func_ndim1(y_pred, y_true):
    return torch.randn(1)


def good_batch_loss_func(y_pred, y_true):
    batch, *_ = y_true.shape
    return torch.randn(batch)


def good_pairwise_loss_func(y_pred, y_true):
    batch, n_src, *_ = y_true.shape
    return torch.randn(batch, n_src, n_src)


@pytest.mark.parametrize("batch_size", [1, 2, 8])
@pytest.mark.parametrize("n_src", [2, 3, 4])
@pytest.mark.parametrize("time", [16000, 1221])
def test_wrapper(batch_size, n_src, time):
    targets = torch.randn(batch_size, n_src, time)
    est_targets = torch.randn(batch_size, n_src, time)
    for bad_loss_func in [bad_loss_func_ndim0, bad_loss_func_ndim1]:
        loss = PITLossWrapper(bad_loss_func)
        with pytest.raises(AssertionError):
            loss(est_targets, targets)
    # wo_src loss function / With and without return estimates
    loss = PITLossWrapper(good_batch_loss_func, pit_from="pw_pt")
    loss(est_targets, targets)
    loss_value, reordered_est = loss(est_targets, targets, return_est=True)
    assert reordered_est.shape == est_targets.shape

    # pairwise loss function / With and without return estimates
    loss = PITLossWrapper(good_pairwise_loss_func, pit_from="pw_mtx")
    loss(est_targets, targets)
    loss_value, reordered_est = loss(est_targets, targets, return_est=True)
    assert reordered_est.shape == est_targets.shape

    # w_src loss function / With and without return estimates
    loss = PITLossWrapper(good_batch_loss_func, pit_from="perm_avg")
    loss(est_targets, targets)
    loss_value, reordered_est = loss(est_targets, targets, return_est=True)
    assert reordered_est.shape == est_targets.shape

    # sinkhorn loss function / With and without return estimates
    loss = PITLossWrapper(good_pairwise_loss_func, pit_from="pw_sinkpit")
    loss(est_targets, targets)
    loss_value, reordered_est = loss(est_targets, targets, return_est=True)
    assert reordered_est.shape == est_targets.shape


@pytest.mark.parametrize(
    "perm",
    list(itertools.permutations([0, 1, 2]))
    + list(itertools.permutations([0, 1, 2, 3]))
    + list(itertools.permutations([0, 1, 2, 3, 4])),
)
def test_permutation(perm):
    """ Construct fake target/estimates pair. Check the value and reordering."""
    n_src = len(perm)
    perm_tensor = torch.Tensor(perm)
    source_base = torch.ones(1, n_src, 10)
    sources = torch.arange(n_src).unsqueeze(-1) * source_base
    est_sources = perm_tensor.unsqueeze(-1) * source_base

    loss_func = PITLossWrapper(pairwise_mse)
    loss_value, reordered = loss_func(est_sources, sources, return_est=True)

    assert loss_value.item() == 0
    assert_allclose(sources, reordered)


def test_permreduce():
    from functools import partial

    n_src = 3
    sources = torch.randn(10, n_src, 8000)
    est_sources = torch.randn(10, n_src, 8000)
    wo_reduce = PITLossWrapper(pairwise_mse, pit_from="pw_mtx")
    w_mean_reduce = PITLossWrapper(
        pairwise_mse,
        pit_from="pw_mtx",
        # perm_reduce=partial(torch.mean, dim=-1))
        perm_reduce=lambda x: torch.mean(x, dim=-1),
    )
    w_sum_reduce = PITLossWrapper(
        pairwise_mse, pit_from="pw_mtx", perm_reduce=partial(torch.sum, dim=-1)
    )

    wo = wo_reduce(est_sources, sources)
    w_mean = w_mean_reduce(est_sources, sources)
    w_sum = w_sum_reduce(est_sources, sources)

    assert_allclose(wo, w_mean)
    assert_allclose(wo, w_sum / n_src)


def test_permreduce_args():
    def reduce_func(perm_losses, class_weights=None):
        # perm_losses is (batch , n_perms, n_src) for now
        if class_weights is None:
            return torch.mean(perm_losses, dim=-1)
        if class_weights.ndim == 2:
            class_weights = class_weights.unsqueeze(1)
        return torch.mean(perm_losses * class_weights, -1)

    n_src = 3
    sources = torch.randn(10, n_src, 8000)
    est_sources = torch.randn(10, n_src, 8000)
    loss_func = PITLossWrapper(pairwise_mse, pit_from="pw_mtx", perm_reduce=reduce_func)
    weights = torch.softmax(torch.randn(10, n_src), dim=-1)
    loss_func(est_sources, sources, reduce_kwargs={"class_weights": weights})


@pytest.mark.parametrize("n_src", [2, 4, 5, 6, 8])
def test_best_perm_match(n_src):
    pwl = torch.randn(2, n_src, n_src)

    min_loss, min_idx = PITLossWrapper.find_best_perm_factorial(pwl)
    min_loss_hun, min_idx_hun = PITLossWrapper.find_best_perm_hungarian(pwl)

    assert_allclose(min_loss, min_loss_hun)
    assert_allclose(min_idx, min_idx_hun)


def test_raises_wrong_pit_from():
    with pytest.raises(ValueError):
        PITLossWrapper(lambda x: x, pit_from="unknown_mode")


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("n_src", [5, 10, 30])
@pytest.mark.parametrize("time", [16000])
@pytest.mark.parametrize("noise_level", [0.1])
def test_sinkhorn_reordering(batch_size, n_src, time, noise_level):
    # random data
    targets = torch.randn(batch_size, n_src, time) * 10  # ground truth
    noise = torch.randn(batch_size, n_src, time) * noise_level
    est_targets = (
        targets[:, torch.randperm(n_src), :] + noise
    )  # reorder channels, and add small noise

    # initialize wrapper
    loss = PITLossWrapper(pairwise_mse, pit_from="pw_sinkpit")

    # check if the reordering is working
    min_loss_valid, reordered = loss(
        est_targets, targets, return_est=True
    )  # beta and n_iter are curretnly hard coded (beta = 100, n_iter = 2000)
    loss_before_reorder = torch.mean((est_targets - targets) ** 2)
    loss_after_reorder = torch.mean((reordered - targets) ** 2)
    noise_energy = torch.mean(noise ** 2)

    # verify that the reordering is correct
    # if correct, `loss_after_reorder` should be very close to the noise energy.
    assert loss_after_reorder <= loss_before_reorder
    assert_allclose(loss_after_reorder, noise_energy)
