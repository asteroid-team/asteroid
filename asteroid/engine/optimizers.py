from torch.optim.optimizer import Optimizer
from torch.optim import Adam, RMSprop, SGD, Adadelta, Adagrad, Adamax, AdamW, ASGD
from torch_optimizer import (
    AccSGD,
    AdaBound,
    AdaMod,
    DiffGrad,
    Lamb,
    NovoGrad,
    PID,
    QHAdam,
    QHM,
    RAdam,
    SGDW,
    Yogi,
    Ranger,
    RangerQH,
    RangerVA,
)


__all__ = [
    "AccSGD",
    "AdaBound",
    "AdaMod",
    "DiffGrad",
    "Lamb",
    "NovoGrad",
    "PID",
    "QHAdam",
    "QHM",
    "RAdam",
    "SGDW",
    "Yogi",
    "Ranger",
    "RangerQH",
    "RangerVA",
    "Adam",
    "RMSprop",
    "SGD",
    "Adadelta",
    "Adagrad",
    "Adamax",
    "AdamW",
    "ASGD",
    "make_optimizer",
    "get",
]


def make_optimizer(params, optimizer="adam", **kwargs):
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


def get(identifier):
    """ Returns an optimizer function from a string. Returns its input if it
    is callable (already a :class:`torch.optim.Optimizer` for example).

    Args:
        identifier (str or Callable): the optimizer identifier.

    Returns:
        :class:`torch.optim.Optimizer` or None
    """
    if isinstance(identifier, Optimizer):
        return identifier
    elif isinstance(identifier, str):
        to_get = {k.lower(): v for k, v in globals().items()}
        cls = to_get.get(identifier.lower())
        if cls is None:
            raise ValueError(f"Could not interpret optimizer : {str(identifier)}")
        return cls
    raise ValueError(f"Could not interpret optimizer : {str(identifier)}")
