import pytest
import torch
from torch.testing import assert_allclose

from asteroid.losses import PITLossWrapper
from asteroid.losses import sdr
from asteroid.losses import nosrc_mse, pairwise_mse, nonpit_mse
from asteroid.losses import deep_clustering_loss
from asteroid.losses.multi_scale_spectral import MultiScaleSpectral


@pytest.mark.parametrize("n_src", [2, 3, 4])
@pytest.mark.parametrize("function_triplet", [
    [sdr.pairwise_neg_sisdr, sdr.nosrc_neg_sisdr, sdr.nonpit_neg_sisdr],
    [sdr.pairwise_neg_sdsdr, sdr.nosrc_neg_sdsdr, sdr.nonpit_neg_sdsdr],
    [sdr.pairwise_neg_snr, sdr.nosrc_neg_snr, sdr.nonpit_neg_snr],
    [pairwise_mse, nosrc_mse, nonpit_mse],
])
def test_sisdr(n_src, function_triplet):
    # Unpack the triplet
    pairwise, nosrc, nonpit = function_triplet
    # Fake targets and estimates
    targets = torch.randn(2, n_src, 32000)
    est_targets = torch.randn(2, n_src, 32000)
    # Create the 3 PIT wrappers
    pw_wrapper = PITLossWrapper(pairwise, mode='pairwise')
    wo_src_wrapper = PITLossWrapper(nosrc, mode='wo_src')
    w_src_wrapper = PITLossWrapper(nonpit, mode='w_src')

    # Circular tests on value
    assert_allclose(pw_wrapper(est_targets, targets),
                    wo_src_wrapper(est_targets, targets))
    assert_allclose(wo_src_wrapper(est_targets, targets),
                    w_src_wrapper(est_targets, targets))

    # Circular tests on returned estimates
    assert_allclose(pw_wrapper(est_targets, targets, return_est=True)[1],
                    wo_src_wrapper(est_targets, targets, return_est=True)[1])
    assert_allclose(wo_src_wrapper(est_targets, targets, return_est=True)[1],
                    w_src_wrapper(est_targets, targets, return_est=True)[1])

@pytest.mark.parametrize("spk_cnt", [2, 3, 4])
def test_dc(spk_cnt):
    embedding = torch.randn(10, 5*400, 20)
    targets = torch.LongTensor(10, 400, 5).random_(0, spk_cnt)
    loss = deep_clustering_loss(embedding, targets, spk_cnt)
    assert loss.shape[0] == 10


@pytest.mark.parametrize("n_src", [2, 3, 4])
def test_multi_scale_spectral_PIT(n_src):
    # Fake targets and estimates
    targets = torch.randn(2, n_src, 32000)
    est_targets = torch.randn(2, n_src, 32000)
    # Create PITLossWrapper in 'wo_src' mode
    loss_func = PITLossWrapper(MultiScaleSpectral(), mode='wo_src')
    # Compute the loss
    loss = loss_func(targets, est_targets)


@pytest.mark.parametrize("batch_size", [2, 3, 4])
def test_multi_scale_spectral_shape(batch_size):
    # Fake targets and estimates
    targets = torch.randn(batch_size, 32000)
    est_targets = torch.randn(batch_size, 32000)
    # Create PITLossWrapper in 'wo_src' mode
    loss_func = MultiScaleSpectral()
    # Compute the loss
    loss = loss_func(targets, est_targets)
    assert loss.shape[0] == batch_size
