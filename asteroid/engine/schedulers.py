from torch.optim.optimizer import Optimizer


class _BaseScheduler(object):
    """ Base class for the step-wise scheduler logic.

    Args:
        optimizer (Optimize): Optimizer instance to apply lr schedule on.
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

    def step(self):
        """Update step-wise learning rate before optimizer.step."""
        self.step_num += 1
        lr = self._get_lr()
        self._set_lr(lr)

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items()}


class NoamScheduler(_BaseScheduler):
    """The Noam learning rate scheduler, originally used in conjunction with
        the Adam optimizer in [1].

    Args:
        optimizer (Optimizer): Optimizer instance to apply lr schedule on.
        d_model(int): The number of units in the layer output.
        warmup_steps (int): The number of steps in the warmup stage of training.
        scale (float): A fixed coefficient for rescaling the final learning rate.

    .. note::
        The Noam scheduler increases the learning rate linearly for the first
        `warmup_steps` steps, and decreases it thereafter proportionally to the
        inverse square root of the step number::
            lr = scale_factor * ( (model_dim ** (-0.5)) * adj_step )
            adj_step = min(step_num ** (-0.5), step_num * warmup_steps ** (-1.5))

    References:
        [1] Vaswani et al. (2017) "Attention is all you need". *31st
           Conference on Neural Information Processing Systems*,
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


class DPTNetScheduler(_BaseScheduler):
    """ Dual Path Transformer Scheduler used in [1]

    Args:
        optimizer (Optimizer): Optimizer instance to apply lr schedule on.
        steps_per_epoch (int): Number of steps per epoch.
        d_model(int): The number of units in the layer output.
        warmup_steps (int): The number of steps in the warmup stage of training.
        noam_scale (float): Linear increase rate in first phase.
        exp_max (float): Max learning rate in second phase.
        exp_base (float): Exp learning rate base in second phase.

    References:
        [1]: Jingjing Chen et al. "Dual-Path Transformer Network: Direct
            Context-Aware Modeling for End-to-End Monaural Speech Separation"
            Interspeech 2020.

    .. note::
        This scheduler increases the learning rate linearly for the first
        `warmup_steps`, and then decay it by 0.98 for every two epochs
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
        self.epoch = None

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
