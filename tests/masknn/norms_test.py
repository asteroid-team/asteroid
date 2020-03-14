import pytest
import torch

from asteroid.masknn import norms


@pytest.mark.parametrize("norm_str", ["gLN", "cLN", "cgLN", "bN"])
@pytest.mark.parametrize("channel_size", [8, 128, 4])
def test_norms(norm_str, channel_size):
    norm_layer = norms.get(norm_str)
    # Use get on the class
    out_from_get = norms.get(norm_layer)
    assert out_from_get == norm_layer
    # Use get on the instance
    norm_layer = norm_layer(channel_size)
    out_from_get = norms.get(norm_layer)
    assert out_from_get == norm_layer

    # Test forward
    inp = torch.randn(4, channel_size, 12)
    _ = norm_layer(inp)


@pytest.mark.parametrize("wrong", ["wrong_string", 12, object()])
def test_get_errors(wrong):
    with pytest.raises(ValueError):
        # Should raise for anything not a Optimizer instance + unknown string
        norms.get(wrong)


def test_get_none():
    assert norms.get(None) is None
