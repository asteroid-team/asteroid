import pytest
import torch
from torch.testing import assert_allclose

from asteroid.losses import PITLossWrapper
from asteroid.losses import nosrc_neg_sisdr, pairwise_neg_sisdr, \
    nonpit_neg_sisdr


@pytest.mark.parametrize("n_src", [2, 3, 4])
def test_sisdr(n_src):
    targets = torch.randn(2, n_src, 32000)
    est_targets = torch.randn(2, n_src, 32000)
    pw_wrapper = PITLossWrapper(pairwise_neg_sisdr, mode='pairwise')
    wo_src_wrapper = PITLossWrapper(nosrc_neg_sisdr, mode='wo_src')
    w_src_wrapper = PITLossWrapper(nonpit_neg_sisdr, mode='w_src')

    pw = pw_wrapper(targets, est_targets)
    wo_src = wo_src_wrapper(targets, est_targets)
    w_src = w_src_wrapper(targets, est_targets)

    assert_allclose(pw_wrapper(targets, est_targets),
                    wo_src_wrapper(targets, est_targets))
    assert_allclose(w_src_wrapper(targets, est_targets),
                    wo_src_wrapper(targets, est_targets))

    assert_allclose(pw_wrapper(targets, est_targets, return_est=True)[1],
                    wo_src_wrapper(targets, est_targets, return_est=True)[1])
    assert_allclose(w_src_wrapper(targets, est_targets, return_est=True)[1],
                    wo_src_wrapper(targets, est_targets, return_est=True)[1])
