import torch
from torch.testing import assert_allclose
from asteroid import filterbanks as fb
from asteroid.filterbanks.inputs_and_masks import _masks


COMPLEX_FBS = [
    fb.STFTFB,
    fb.ParamSincFB,
    fb.AnalyticFreeFB
]


def make_encoder_from(fb_class, config):
    enc = fb.Encoder(fb_class(**config))
    fb_dim = enc.filterbank.n_feats_out
    return enc, fb_dim


def test_output_dims():
    fb_config = {
        'n_filters': 512,
        'kernel_size': 256,
        'stride': 128
    }
    for fb_class in COMPLEX_FBS:
        enc, fb_dim = make_encoder_from(fb_class, fb_config)
        tf_rep = enc(torch.randn(2, 1, 16000)) # [batch, freq, time]
        assert fb_dim % 2 == 0
        assert tf_rep.shape[1] == fb_dim


def test_mag_mask():
    fb_config = {
        'n_filters': 512,
        'kernel_size': 256,
        'stride': 128
    }
    for fb_class in COMPLEX_FBS:
        enc, fb_dim = make_encoder_from(fb_class, fb_config)
        tf_rep = enc(torch.randn(2, 1, 16000))  # [batch, freq, time]
        id_mag_mask = torch.ones((1, fb_dim//2, 1))
        assert_allclose(_masks['mag'][0](tf_rep, id_mag_mask, dim=1), tf_rep)


def test_reim_mask():
    fb_config = {
        'n_filters': 512,
        'kernel_size': 256,
        'stride': 128
    }
    for fb_class in COMPLEX_FBS:
        enc, fb_dim = make_encoder_from(fb_class, fb_config)
        tf_rep = enc(torch.randn(2, 1, 16000))  # [batch, freq, time]
        id_reim_mask = torch.ones((1, fb_dim, 1))
        assert_allclose(_masks['reim'][0](tf_rep, id_reim_mask, dim=1),
                        tf_rep)


def test_comp_mask():
    fb_config = {
        'n_filters': 512,
        'kernel_size': 256,
        'stride': 128
    }
    for fb_class in COMPLEX_FBS:
        enc, fb_dim = make_encoder_from(fb_class, fb_config)
        tf_rep = enc(torch.randn(2, 1, 16000))  # [batch, freq, time]
        id_complex_mask = torch.cat((torch.ones((1, fb_dim // 2, 1)),
                                     torch.zeros((1, fb_dim // 2, 1))), dim=1)
        assert_allclose(_masks['comp'][0](tf_rep, id_complex_mask, dim=1),
                        tf_rep)
