from torch import optim


def make_optimizer(params, optimizer='adam', **kwargs):
    return get(optimizer)(params, **kwargs)


def adam(params, lr=0.001, **kwargs):
    return optim.Adam(params, lr=lr, **kwargs)


def sgd(params, lr=0.001, **kwargs):
    return optim.SGD(params, lr=lr, **kwargs)


def rmsprop(params, lr=0.001, **kwargs):
    return optim.RMSprop(params, lr=lr, **kwargs)


def get(identifier):
    if identifier is None:
        return None
    elif isinstance(identifier, optim.Optimizer):
        return identifier
    elif isinstance(identifier, str):
        cls = globals().get(identifier)
        if cls is None:
            raise ValueError('Could not interpret optimizer identifier: ' +
                             str(identifier))
        return cls
    else:
        raise ValueError('Could not interpret optimizer identifier: ' +
                         str(identifier))
