import pytest
import torch
from torch.testing import assert_allclose

from asteroid.losses import PITLossWrapper
from asteroid.losses import sdr


@pytest.mark.parametrize("n_src", [2, 3, 4])
@pytest.mark.parametrize("function_triplet", [
    [sdr.pairwise_neg_sisdr, sdr.nosrc_neg_sisdr, sdr.nonpit_neg_sisdr],
    [sdr.pairwise_neg_sdsdr, sdr.nosrc_neg_sdsdr, sdr.nonpit_neg_sdsdr],
    [sdr.pairwise_neg_snr, sdr.nosrc_neg_snr, sdr.nonpit_neg_snr],
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
