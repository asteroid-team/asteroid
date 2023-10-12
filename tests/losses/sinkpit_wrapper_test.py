import pytest
import itertools
import torch
from torch import nn, optim
from torch.utils import data
from torch.testing import assert_close

import pytorch_lightning as pl
from pytorch_lightning import Trainer

from asteroid.losses import PITLossWrapper
from asteroid.losses import sdr
from asteroid.losses import singlesrc_mse, pairwise_mse, multisrc_mse
from asteroid.engine.system import System
from asteroid.utils.test_utils import DummyWaveformDataset

# target modules
from asteroid.losses import SinkPITLossWrapper
from asteroid.engine.schedulers import sinkpit_default_beta_schedule, SinkPITBetaScheduler


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
@pytest.mark.parametrize("n_src", [2, 5, 10])
@pytest.mark.parametrize("time", [16000, 1221])
def test_wrapper(batch_size, n_src, time):
    targets = torch.randn(batch_size, n_src, time)
    est_targets = torch.randn(batch_size, n_src, time)
    for bad_loss_func in [bad_loss_func_ndim0, bad_loss_func_ndim1]:
        loss = SinkPITLossWrapper(bad_loss_func)
        with pytest.raises(AssertionError):
            loss(est_targets, targets)

    loss = SinkPITLossWrapper(good_pairwise_loss_func)
    loss(est_targets, targets)
    loss_value, reordered_est = loss(est_targets, targets, return_est=True)
    assert reordered_est.shape == est_targets.shape


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("n_src", [2, 3, 4])
@pytest.mark.parametrize("beta,n_iter", [(100.0, 2000)])
@pytest.mark.parametrize(
    "function_triplet",
    [
        [sdr.pairwise_neg_sisdr, sdr.singlesrc_neg_sisdr, sdr.multisrc_neg_sisdr],
        [sdr.pairwise_neg_sdsdr, sdr.singlesrc_neg_sdsdr, sdr.multisrc_neg_sdsdr],
        [sdr.pairwise_neg_snr, sdr.singlesrc_neg_snr, sdr.multisrc_neg_snr],
        [pairwise_mse, singlesrc_mse, multisrc_mse],
    ],
)
def test_proximity_sinkhorn_hungarian(batch_size, n_src, beta, n_iter, function_triplet):
    time = 16000
    noise_level = 0.1
    pairwise, nosrc, nonpit = function_triplet

    # random data
    targets = torch.randn(batch_size, n_src, time) * 10  # ground truth
    noise = torch.randn(batch_size, n_src, time) * noise_level
    est_targets = (
        targets[:, torch.randperm(n_src), :] + noise
    )  # reorder channels, and add small noise

    # initialize wrappers
    loss_sinkhorn = SinkPITLossWrapper(pairwise, n_iter=n_iter)
    loss_hungarian = PITLossWrapper(pairwise, pit_from="pw_mtx")

    # compute loss by sinkhorn
    loss_sinkhorn.beta = beta
    mean_loss_sinkhorn = loss_sinkhorn(est_targets, targets, return_est=False)

    # compute loss by hungarian
    mean_loss_hungarian = loss_hungarian(est_targets, targets, return_est=False)

    # compare
    assert_close(mean_loss_sinkhorn, mean_loss_hungarian)


class _TestCallback(pl.callbacks.Callback):
    def __init__(self, function, total, batch_size):
        self.f = function
        self.epoch = 0
        self.n_batch = total // batch_size

    def on_train_batch_end(self, trainer, *args, **kwargs):
        step = trainer.global_step
        assert self.epoch * self.n_batch <= step
        assert step <= (self.epoch + 1) * self.n_batch

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        assert epoch == self.epoch
        assert pl_module.loss_func.beta == self.f(epoch)
        self.epoch += 1


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("n_src", [2, 10])
@pytest.mark.parametrize("len_wave", [100])
@pytest.mark.parametrize(
    "beta_schedule",
    [
        sinkpit_default_beta_schedule,  # default
        lambda epoch: 123.0 if epoch < 3 else 456.0,  # test if lambda function works
    ],
)
def test_sinkpit_beta_scheduler(batch_size, n_src, len_wave, beta_schedule):
    model = nn.Sequential(nn.Conv1d(1, n_src, 1), nn.ReLU())
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    dataset = DummyWaveformDataset(total=2 * batch_size, n_src=n_src, len_wave=len_wave)
    loader = data.DataLoader(
        dataset, batch_size=batch_size, num_workers=0
    )  # num_workers=0 means doing everything in the main process without calling subprocesses

    system = System(
        model,
        optimizer,
        loss_func=SinkPITLossWrapper(sdr.pairwise_neg_sisdr, n_iter=5),
        train_loader=loader,
        val_loader=loader,
    )

    trainer = pl.Trainer(
        max_epochs=10,
        fast_dev_run=False,
        callbacks=[
            SinkPITBetaScheduler(beta_schedule),
            _TestCallback(
                beta_schedule, len(dataset), batch_size
            ),  # test if beta are the same at epoch_start and epoch_end.
        ],
    )

    trainer.fit(system)
