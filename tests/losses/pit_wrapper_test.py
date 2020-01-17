import pytest
import torch
from asteroid.losses import PITLossWrapper


def bad_loss_func_ndim0(y_true, y_pred):
    return torch.randn(1).mean()


def bad_loss_func_ndim1(y_true, y_pred):
    return torch.randn(1)


def good_batch_loss_func(y_true, y_pred):
    batch, *_ = y_true.shape
    return torch.randn(batch)


def good_pairwise_loss_func(y_true, y_pred):
    batch, n_src, *_ = y_true.shape
    return torch.randn(batch, n_src, n_src)


@pytest.mark.parametrize("batch_size", [1, 2, 8, 16])
@pytest.mark.parametrize("n_src", [2, 3, 4])
@pytest.mark.parametrize("time", [16000, 32000, 1221])
def test_wrapper(batch_size, n_src, time):
    targets = torch.randn(batch_size, n_src, time)
    est_targets = torch.randn(batch_size, n_src, time)
    for bad_loss_func in [bad_loss_func_ndim0, bad_loss_func_ndim1]:
        loss = PITLossWrapper(bad_loss_func)
        with pytest.raises(AssertionError):
            loss(targets, est_targets)
    # wo_src loss function / With and without return estimates
    loss = PITLossWrapper(good_batch_loss_func, mode='wo_src')
    loss_value_no_return = loss(targets, est_targets)
    loss_value, reordered_est = loss(targets, est_targets, return_est=True)
    assert reordered_est.shape == est_targets.shape

    # pairwise loss function / With and without return estimates
    loss = PITLossWrapper(good_pairwise_loss_func, mode='pairwise')
    loss_value_no_return = loss(targets, est_targets)
    loss_value, reordered_est = loss(targets, est_targets, return_est=True)
    assert reordered_est.shape == est_targets.shape

    # w_src loss function / With and without return estimates
    loss = PITLossWrapper(good_batch_loss_func, mode='w_src')
    loss_value_no_return = loss(targets, est_targets)
    loss_value, reordered_est = loss(targets, est_targets, return_est=True)
    assert reordered_est.shape == est_targets.shape



