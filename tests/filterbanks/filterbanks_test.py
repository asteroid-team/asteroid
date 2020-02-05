import pytest
import torch
from torch.testing import assert_allclose

from asteroid import filterbanks
from asteroid.filterbanks import Encoder, Decoder
from asteroid.filterbanks import FreeFB, AnalyticFreeFB, ParamSincFB
from asteroid.filterbanks import make_enc_dec


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
    # Definition
    enc = Encoder(fb_class(**fb_config))
    dec = Decoder(fb_class(**fb_config))
    # Forward
    inp = torch.randn(1, 1, 32000)
    tf_out = enc(inp)
    out = dec(tf_out)
    # 4d forward + unit test
    tf_out_4d = tf_out.repeat(1, 2, 1, 1)
    out_4d = dec(tf_out_4d)
    assert_allclose(out, out_4d[:, 0])
    # Get config tests
    dec_config = dec.get_config()
    enc_config = enc.get_config()
    # N feats out test
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


@pytest.mark.parametrize("fb_class", [FreeFB, AnalyticFreeFB, ParamSincFB])
def test_pinv_of(fb_class):
    fb = fb_class(n_filters=500, kernel_size=16, stride=8)
    encoder = Encoder(fb)
    # Pseudo inverse can be taken from an Encoder/Decoder class or Filterbank.
    decoder_e = Decoder.pinv_of(encoder)
    decoder_f = Decoder.pinv_of(fb)
    assert_allclose(decoder_e.filters, decoder_f.filters)

    # Check filter computing
    inp = torch.randn(1, 1, 32000)
    _ = decoder_e(encoder(inp))

    decoder = Decoder(fb)
    # Pseudo inverse can be taken from an Encoder/Decoder class or Filterbank.
    encoder_e = Encoder.pinv_of(decoder)
    encoder_f = Encoder.pinv_of(fb)
    assert_allclose(encoder_e.filters, encoder_f.filters)


@pytest.mark.parametrize("who", ["enc", "dec"])
def test_make_enc_dec(who):
    fb_config = {"n_filters": 500,
                 "kernel_size": 16,
                 "stride": 8}
    enc, dec = make_enc_dec("free", who_is_pinv=who, **fb_config)
    enc, dec = make_enc_dec(FreeFB, who_is_pinv=who, **fb_config)
    assert enc.filterbank == filterbanks.get(enc.filterbank)


@pytest.mark.parametrize("wrong", ["wrong_string", 12, object()])
def test_get_errors(wrong):
    with pytest.raises(ValueError):
        # Should raise for anything not a Optimizer instance + unknown string
        filterbanks.get(wrong)


def test_get_none():
    assert filterbanks.get(None) is None
