

class _BaseScheduler(object):
    def __init__(self):
        self.step_num = 1

    @staticmethod
    def _get_lr(self):
        raise NotImplementedError

    @staticmethod
    def _set_lr(self, lr):
        for param_group in self.param_groups:
            param_group['lr'] = lr

    @staticmethod
    def step(self):
        self.step_num += 1
        lr = self._get_lr(self)
        self._set_lr(self, lr)
        self.step(self)

    #def load_state_dict(self, state_dict):
     #   self.__dict__.update(state_dict)

    #def state_dict(self):
     #   return {key: value for key, value in self.__dict__.items() if key != "optimizer"}


class NoamScheduler(_BaseScheduler):
    def __init__(self, d_model, warmup_steps, scale=1.0):
        super().__init__()
        self.d_model = d_model
        self.scale = scale
        self.warmup_steps = warmup_steps

    @staticmethod
    def _get_lr(self):
        lr = self.scale * self.d_model ** (-0.5) * \
                min(self.step_num ** (-0.5), self.step_num * self.warmup_steps ** (-1.5))
        return lr


class DPTNetScheduler(_BaseScheduler):
    def __init__(self, steps_per_epoch, d_model, warmup_steps=4000, noam_scale=1.0,
                 exp_max=0.0004, exp_base=0.98):
        super().__init__()
        self.noam_scale = noam_scale
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.exp_max = exp_max
        self.exp_base = exp_base
        self.steps_per_epoch = steps_per_epoch
        self.epoch = 0

    @staticmethod
    def _get_lr(self):
        if self.step_num % self.steps_per_epoch == 0:
            self.epoch += 1

        if self.step_num > self.warmup_steps:
            # exp decaying
            lr = self.exp_max * (self.exp_base ** ((self.epoch - 1) // 2))
        else:
            # noam
            lr = self.noam_scale * self.d_model ** (-0.5) * \
                min(self.step_num ** (-0.5), self.step_num * self.warmup_steps ** (-1.5))
        return lr

    def apply(self, optimizer):
        optimizer.__dict__.update(self.__dict__)
        optimizer._get_lr = DPTNetScheduler._get_lr(optimizer)
        optimizer._set_lr = DPTNetScheduler._set_lr(optimizer, optimizer._get_lr())
        optimizer.step = DPTNetScheduler.step(optimizer)
        return optimizer
