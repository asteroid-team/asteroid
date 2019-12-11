"""
| Activation functions which are retrievable from a string (all lower-case).
| @author : Manuel Pariente, Inria-Nancy
"""

from torch import nn


def linear():
    return LinearActivation()


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


class LinearActivation(nn.Module):
    def forward(self, x):
        return x


def get(identifier):
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
