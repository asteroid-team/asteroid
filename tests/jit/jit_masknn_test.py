import pytest
import torch
from torch.testing import assert_close
from asteroid.masknn import norms


@pytest.mark.parametrize("cls", (norms.GlobLN, norms.FeatsGlobLN, norms.ChanLN))
def test_lns(cls):
    chan_size = 10
    model = cls(channel_size=chan_size)
    x = torch.randn(1, chan_size, 12)

    traced = torch.jit.trace(model, x)

    y = torch.randn(3, chan_size, 18, 12)
    assert_close(traced(y), model(y))

    y = torch.randn(2, chan_size, 10, 5, 4)
    assert_close(traced(y), model(y))


def test_cumln():
    chan_size = 10
    model = norms.CumLN(channel_size=chan_size)
    x = torch.randn(1, chan_size, 12)

    traced = torch.jit.trace(model, x)

    y = torch.randn(3, chan_size, 100)
    assert_close(traced(y), model(y))
