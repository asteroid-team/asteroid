import pytest
import torch

from asteroid.filterbanks import Encoder, Decoder
from asteroid.filterbanks import FreeFB, AnalyticFreeFB, ParamSincFB


def fb_config_list():
    keys = ['n_filters', 'kernel_size', 'stride']
    param_list = [
        [512, 256, 128],
        [512, 256, 64],
        [512, 32, 16],
        [500, 16, 8],
        [200, 80, 80],
        [200, 80, 20],
    ]
    return [dict(zip(keys, values)) for values in param_list]


@pytest.mark.parametrize("fb_class", [FreeFB, AnalyticFreeFB, ParamSincFB])
@pytest.mark.parametrize("fb_config", fb_config_list())
def test_fb_def_and_forward(fb_class, fb_config):
    """ Test filterbank defintion and encoder/decoder forward."""
    enc = Encoder(fb_class(**fb_config))
    dec = Decoder(fb_class(**fb_config))
    inp = torch.randn(1, 1, 32000)
    tf_out = enc(inp)
    out = dec(tf_out)
    assert tf_out.shape[1] == enc.filterbank.n_feats_out


@pytest.mark.parametrize("fb_class", [AnalyticFreeFB, ParamSincFB])
@pytest.mark.parametrize("n_filters", [256, 257])
@pytest.mark.parametrize("kernel_size", [256, 257])
def test_complexfb_shapes(fb_class, n_filters, kernel_size):
    fb = fb_class(n_filters, kernel_size)
    assert fb.filters.shape[0] == 2 * (n_filters // 2)


@pytest.mark.parametrize("kernel_size", [256, 257, 128, 129])
def test_paramsinc_shape(kernel_size):
    """ ParamSincFB has odd length filters """
    fb = ParamSincFB(n_filters=200, kernel_size=kernel_size)
    assert fb.filters.shape[-1] == 2 * (kernel_size // 2) + 1
