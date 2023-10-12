import torch
from torch.optim.optimizer import Optimizer
import pytorch_lightning as pl

from ..losses import SinkPITLossWrapper


class BaseScheduler(object):
    """Base class for the step-wise scheduler logic.

    Args:
        optimizer (Optimize): Optimizer instance to apply lr schedule on.

    Subclass this and overwrite ``_get_lr`` to write your own step-wise scheduler.
    """

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.step_num = 0

    def zero_grad(self):
        self.optimizer.zero_grad()

    def _get_lr(self):
        raise NotImplementedError

    def _set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def step(self, metrics=None, epoch=None):
        """Update step-wise learning rate before optimizer.step."""
        self.step_num += 1
        lr = self._get_lr()
        self._set_lr(lr)

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != "optimizer"}

    def as_tensor(self, start=0, stop=100_000):
        """Returns the scheduler values from start to stop."""
        lr_list = []
        for _ in range(start, stop):
            self.step_num += 1
            lr_list.append(self._get_lr())
        self.step_num = 0
        return torch.tensor(lr_list)

    def plot(self, start=0, stop=100_000):  # noqa
        """Plot the scheduler values from start to stop."""
        import matplotlib.pyplot as plt

        all_lr = self.as_tensor(start=start, stop=stop)
        plt.plot(all_lr.numpy())
        plt.show()


class NoamScheduler(BaseScheduler):
    r"""The Noam learning rate scheduler, originally used in conjunction with
    the Adam optimizer in [1].

    Args:
        optimizer (Optimizer): Optimizer instance to apply lr schedule on.
        d_model(int): The number of units in the layer output.
        warmup_steps (int): The number of steps in the warmup stage of training.
        scale (float): A fixed coefficient for rescaling the final learning rate.

    Schedule:
        The Noam scheduler increases the learning rate linearly for the first
        ``warmup_steps`` steps, and decreases it thereafter proportionally to the
        inverse square root of the step number:
            :math:`lr = scale\_factor * ( model\_dim^{-0.5} * adj\_step )`
            :math:`adj\_step = min(step\_num^{0.5}, step\_num * warmup\_steps^{-1.5})`

    References
        [1] Vaswani et al. (2017) "Attention is all you need". 31st
        Conference on Neural Information Processing Systems
    """

    def __init__(self, optimizer, d_model, warmup_steps, scale=1.0):
        super().__init__(optimizer)
        self.d_model = d_model
        self.scale = scale
        self.warmup_steps = warmup_steps

    def _get_lr(self):
        lr = (
            self.scale
            * self.d_model ** (-0.5)
            * min(self.step_num ** (-0.5), self.step_num * self.warmup_steps ** (-1.5))
        )
        return lr


class DPTNetScheduler(BaseScheduler):
    """Dual Path Transformer Scheduler used in [1]

    Args:
        optimizer (Optimizer): Optimizer instance to apply lr schedule on.
        steps_per_epoch (int): Number of steps per epoch.
        d_model(int): The number of units in the layer output.
        warmup_steps (int): The number of steps in the warmup stage of training.
        noam_scale (float): Linear increase rate in first phase.
        exp_max (float): Max learning rate in second phase.
        exp_base (float): Exp learning rate base in second phase.

    Schedule:
        This scheduler increases the learning rate linearly for the first
        ``warmup_steps``, and then decay it by 0.98 for every two epochs.

    References
        [1]: Jingjing Chen et al. "Dual-Path Transformer Network: Direct Context-
        Aware Modeling for End-to-End Monaural Speech Separation" Interspeech 2020.
    """

    def __init__(
        self,
        optimizer,
        steps_per_epoch,
        d_model,
        warmup_steps=4000,
        noam_scale=1.0,
        exp_max=0.0004,
        exp_base=0.98,
    ):
        super().__init__(optimizer)
        self.noam_scale = noam_scale
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.exp_max = exp_max
        self.exp_base = exp_base
        self.steps_per_epoch = steps_per_epoch
        self.epoch = 0

    def _get_lr(self):
        if self.step_num % self.steps_per_epoch == 0:
            self.epoch += 1

        if self.step_num > self.warmup_steps:
            # exp decaying
            lr = self.exp_max * (self.exp_base ** ((self.epoch - 1) // 2))
        else:
            # noam
            lr = (
                self.noam_scale
                * self.d_model ** (-0.5)
                * min(self.step_num ** (-0.5), self.step_num * self.warmup_steps ** (-1.5))
            )
        return lr


def sinkpit_default_beta_schedule(epoch):
    return min([1.02**epoch, 10])


class SinkPITBetaScheduler(pl.callbacks.Callback):
    r"""Scheduler of the beta value of SinkPITLossWrapper
    This module is used as a Callback function of `pytorch_lightning.Trainer`.

    Args:
        cooling_schedule (callable) : A callable that takes a parameter `epoch` (int)
            and returns the value of `beta` (float).

    The default function is ``sinkpit_default_beta_schedule``: :math:`\beta = min(1.02^{epoch}, 10)`

    Example
        >>> from pytorch_lightning import Trainer
        >>> from asteroid.losses import SinkPITBetaScheduler
        >>> # Default scheduling function
        >>> sinkpit_beta_schedule = SinkPITBetaSchedule()
        >>> trainer = Trainer(callbacks=[sinkpit_beta_schedule])
        >>> # User-defined schedule
        >>> sinkpit_beta_schedule = SinkPITBetaScheduler(lambda ep: 1. if ep < 10 else 100.)
        >>> trainer = Trainer(callbacks=[sinkpit_beta_schedule])
    """

    def __init__(self, cooling_schedule=sinkpit_default_beta_schedule):
        self.cooling_schedule = cooling_schedule

    def on_train_epoch_start(self, trainer, pl_module):
        assert isinstance(pl_module.loss_func, SinkPITLossWrapper)
        assert trainer.current_epoch == pl_module.current_epoch  # same
        epoch = pl_module.current_epoch
        # step = pl_module.global_step
        beta = self.cooling_schedule(epoch)
        pl_module.loss_func.beta = beta


# Backward compat
_BaseScheduler = BaseScheduler
