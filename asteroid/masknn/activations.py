from torch import nn


def linear():
    return nn.Identity()


def relu():
    return nn.ReLU()


def prelu():
    return nn.PReLU()


def leaky_relu():
    return nn.LeakyReLU()


def sigmoid():
    return nn.Sigmoid()


def softmax(dim=None):
    return nn.Softmax(dim=dim)


def tanh():
    return nn.Tanh()


def get(identifier):
    """ Returns an activation function from a string. Returns its input if it
    is callable (already an activation for example).

    Args:
        identifier (str or Callable or None): the activation identifier.

    Returns:
        :class:`nn.Module` or None
    """
    if identifier is None:
        return None
    elif callable(identifier):
        return identifier
    elif isinstance(identifier, str):
        cls = globals().get(identifier)
        if cls is None:
            raise ValueError('Could not interpret activation identifier: ' +
                             str(identifier))
        return cls
    else:
        raise ValueError('Could not interpret activation identifier: ' +
                         str(identifier))
