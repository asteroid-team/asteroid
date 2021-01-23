from functools import partial
import torch
from torch import nn
from .. import complex_nn


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


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


def gelu():
    return nn.GELU()


def swish():
    return Swish()


def register_activation(custom_act):
    """Register a custom activation, gettable with `activation.get`.

    Args:
        custom_act: Custom activation function to register.

    """
    if custom_act.__name__ in globals().keys() or custom_act.__name__.lower() in globals().keys():
        raise ValueError(f"Activation {custom_act.__name__} already exists. Choose another name.")
    globals().update({custom_act.__name__: custom_act})


def get(identifier):
    """Returns an activation function from a string. Returns its input if it
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
            raise ValueError("Could not interpret activation identifier: " + str(identifier))
        return cls
    else:
        raise ValueError("Could not interpret activation identifier: " + str(identifier))


def get_complex(identifier):
    """Like `.get` but returns a complex activation created with `asteroid.complex_nn.OnReIm`."""
    activation = get(identifier)
    if activation is None:
        return None
    else:
        return partial(complex_nn.OnReIm, activation)
