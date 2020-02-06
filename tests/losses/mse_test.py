import pytest
import torch
from torch.testing import assert_allclose

from asteroid.losses import PITLossWrapper
from asteroid.losses import nosrc_mse, pairwise_mse, nonpit_mse


@pytest.mark.parametrize("n_src", [2, 3, 4, 5])
def test_mse(n_src):
    targets = torch.randn(2, n_src, 32000)
    est_targets = torch.randn(2, n_src, 32000)
    pw_wrapper = PITLossWrapper(pairwise_mse, mode='pairwise')
    wo_src_wrapper = PITLossWrapper(nosrc_mse, mode='wo_src')
    w_src_wrapper = PITLossWrapper(nonpit_mse, mode='w_src')

    pw = pw_wrapper(est_targets, targets)
    wo_src = wo_src_wrapper(est_targets, targets)
    w_src = w_src_wrapper(est_targets, targets)

    assert_allclose(pw_wrapper(est_targets, targets),
                    wo_src_wrapper(est_targets, targets))
    assert_allclose(w_src_wrapper(est_targets, targets),
                    wo_src_wrapper(est_targets, targets))

    assert_allclose(pw_wrapper(est_targets, targets, return_est=True)[1],
                    wo_src_wrapper(est_targets, targets, return_est=True)[1])
    assert_allclose(w_src_wrapper(est_targets, targets, return_est=True)[1],
                    wo_src_wrapper(est_targets, targets, return_est=True)[1])



