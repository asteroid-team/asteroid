# Call activations and get the same output from torch?
# list of strings / function pair in parametrize?
import pytest
import torch
from torch.testing import assert_close

from asteroid.masknn import activations
from torch import nn


def activation_mapping():
    mapping_list = [
        (nn.Identity, "linear"),
        (nn.ReLU, "relu"),
        (nn.PReLU, "prelu"),
        (nn.LeakyReLU, "leaky_relu"),
        (nn.Sigmoid, "sigmoid"),
        (nn.Tanh, "tanh"),
    ]
    return mapping_list


@pytest.mark.parametrize("activation_tuple", activation_mapping())
def test_activations(activation_tuple):
    torch_act, asteroid_act = activation_tuple
    torch_act = torch_act()
    asteroid_act = activations.get(asteroid_act)()

    inp = torch.randn(10, 11, 12)
    assert_close(torch_act(inp), asteroid_act(inp))


def test_softmax():
    torch_softmax = nn.Softmax(dim=-1)
    asteroid_softmax = activations.get("softmax")(dim=-1)
    inp = torch.randn(10, 11, 12)
    assert_close(torch_softmax(inp), asteroid_softmax(inp))
    assert torch_softmax == activations.get(torch_softmax)


@pytest.mark.parametrize("wrong", ["wrong_string", 12, object()])
def test_get_errors(wrong):
    with pytest.raises(ValueError):
        # Should raise for anything not a Optimizer instance + unknown string
        activations.get(wrong)


def test_get_none():
    assert activations.get(None) is None


def test_register():
    class Custom(nn.Module):
        def __init__(self):
            super().__init__()

    activations.register_activation(Custom)
    cls = activations.get("Custom")
    assert cls == Custom

    with pytest.raises(ValueError):
        activations.register_activation(activations.relu)
