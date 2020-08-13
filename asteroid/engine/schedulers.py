
class _BaseScheduler(object):
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.step_num = 0

    def zero_grad(self):
        self.optimizer.zero_grad()

    def _get_lr(self):
        raise NotImplementedError

    def _set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def step(self):
        self.step_num += 1
        lr = self._get_lr()
        self._set_lr(lr)
        self.optimizer.step()

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items()}



class NoamScheduler(_BaseScheduler):
    def __init__(self, optimizer, d_model, warmup_steps, scale=1.0):
        super(NoamScheduler, self).__init__(optimizer)
        self.d_model = d_model
        self.scale = scale
        self.warmup_steps = warmup_steps

    def _get_lr(self):
        lr = self.scale * self.d_model ** (-0.5) * \
                min(self.step_num ** (-0.5), self.step_num * self.warmup_steps ** (-1.5))
        return lr


class DPTNetScheduler(_BaseScheduler):
    def __init__(self, optimizer, steps_per_epoch, d_model, warmup_steps, scale=1.0):
        super(DPTNetScheduler, self).__init__(optimizer)
        self.scale = scale
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.steps_per_epoch = steps_per_epoch
        self.epoch = None

    def _get_lr(self):
        if self.step_num % self.steps_per_epoch == 0:
            self.epoch += 1

        if self.step_num > self.warmup_steps:
            # exp decaying
            lr = 0.0004 * (0.98 ** ((self.epoch - 1) // 2))
        else:
            # noam
            lr = self.scale * self.d_model ** (-0.5) * \
                min(self.step_num ** (-0.5), self.step_num * self.warmup_steps ** (-1.5))
        return lr

