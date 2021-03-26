import pytest
import torch
from asteroid.masknn import TDConvNet, TDConvNetpp


@pytest.mark.parametrize("mask_act", ["relu", "softmax"])
@pytest.mark.parametrize("out_chan", [None, 10])
@pytest.mark.parametrize("skip_chan", [0, 12])
@pytest.mark.parametrize("causal", [True, False])
def test_tdconvnet(mask_act, out_chan, skip_chan, causal):
    in_chan, n_src = 20, 2
    model = TDConvNet(
        in_chan=in_chan,
        n_src=n_src,
        mask_act=mask_act,
        n_blocks=2,
        n_repeats=2,
        bn_chan=10,
        hid_chan=11,
        skip_chan=skip_chan,
        out_chan=out_chan,
        causal=causal,
    )
    batch, n_frames = 2, 24
    inp = torch.randn(batch, in_chan, n_frames)
    out = model(inp)
    _ = model.get_config()
    out_chan = out_chan if out_chan else in_chan
    assert out.shape == (batch, n_src, out_chan, n_frames)


@pytest.mark.parametrize("mask_act", ["relu", "softmax"])
@pytest.mark.parametrize("out_chan", [None, 10])
@pytest.mark.parametrize("skip_chan", [0, 12])
def test_tdconvnetpp(mask_act, out_chan, skip_chan):
    in_chan, n_src = 20, 2
    model = TDConvNetpp(
        in_chan=in_chan,
        n_src=n_src,
        mask_act=mask_act,
        n_blocks=2,
        n_repeats=2,
        bn_chan=10,
        hid_chan=11,
        skip_chan=skip_chan,
        out_chan=out_chan,
    )
    batch, n_frames = 2, 24
    inp = torch.randn(batch, in_chan, n_frames)
    out, consistency_weights = model(inp)
    _ = model.get_config()
    out_chan = out_chan if out_chan else in_chan
    assert out.shape == (batch, n_src, out_chan, n_frames)
