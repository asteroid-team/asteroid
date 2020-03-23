from torch import optim


def make_optimizer(params, optimizer='adam', **kwargs):
    """

    Args:
        params (iterable): Output of `nn.Module.parameters()`.
        optimizer (str or :class:`torch.optim.Optimizer`): Identifier understood
            by :func:`~.get`.
        **kwargs (dict): keyword arguments for the optimizer.

    Returns:
        torch.optim.Optimizer
    Examples:
        >>> from torch import nn
        >>> model = nn.Sequential(nn.Linear(10, 10))
        >>> optimizer = make_optimizer(model.parameters(), optimizer='sgd',
        >>>                            lr=1e-3)
    """
    return get(optimizer)(params, **kwargs)


def adam(params, lr=0.001, **kwargs):
    return optim.Adam(params, lr=lr, **kwargs)


def sgd(params, lr=0.001, **kwargs):
    return optim.SGD(params, lr=lr, **kwargs)


def rmsprop(params, lr=0.001, **kwargs):
    return optim.RMSprop(params, lr=lr, **kwargs)


def ranger(params, lr=0.001, **kwargs):
    from asranger import Ranger
    return Ranger(params, lr=lr, **kwargs)


def get(identifier):
    """ Returns an optimizer function from a string. Returns its input if it
    is callable (already a :class:`torch.optim.Optimizer` for example).

    Args:
        identifier (str or Callable or None): the optimizer identifier.

    Returns:
        :class:`torch.optim.Optimizer` or None
    """
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
