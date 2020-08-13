
class BaseScheduler(object):
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.step_num = 0

    def zero_grad(self):
        self.optimizer.zero_grad()

    def _update_lr(self):
        self.step_num += 1
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] # change here your lr
        raise NotImplementedError

    def step(self):
        self._update_lr()
        self.optimizer.step()

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items()}
