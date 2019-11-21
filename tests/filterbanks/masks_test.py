import pytest
import torch
from torch.testing import assert_allclose

from asteroid import filterbanks as fb
from asteroid.filterbanks.inputs_and_masks import _masks


COMPLEX_FBS = [
    fb.STFTFB,
    fb.ParamSincFB,
    fb.AnalyticFreeFB
]


@pytest.fixture(scope="module")
def fb_config_list():
    keys = ['n_filters', 'kernel_size', 'stride']
    param_list = [
        [512, 256, 128],
        [512, 256, 64],
        [512, 32, 16],
        [512, 16, 8],
        [512, 257, 64],
        [512, 80, 40],
        [513, 80, 40],
        [514, 80, 40]
    ]
    return [dict(zip(keys, values)) for values in param_list]


@pytest.fixture(scope="module")
def encoder_list(fb_config_list):
    enc_list = []
    for fb_class in COMPLEX_FBS:
        for fb_config in fb_config_list:
            enc_list.append(make_encoder_from(fb_class, fb_config))
    return enc_list


def make_encoder_from(fb_class, config):
    enc = fb.Encoder(fb_class(**config))
    fb_dim = enc.filterbank.n_feats_out
    return enc, fb_dim


def test_output_dims(encoder_list):
    for (enc, fb_dim) in encoder_list:
        tf_rep = enc(torch.randn(2, 1, 16000))  # [batch, freq, time]
        assert fb_dim % 2 == 0
        assert tf_rep.shape[1] == fb_dim


def test_mag_mask(encoder_list):
    for (enc, fb_dim) in encoder_list:
        tf_rep = enc(torch.randn(2, 1, 16000))  # [batch, freq, time]
        id_mag_mask = torch.ones((1, fb_dim//2, 1))
        assert_allclose(_masks['mag'][0](tf_rep, id_mag_mask, dim=1),
                        tf_rep)


def test_reim_mask(encoder_list):
    for (enc, fb_dim) in encoder_list:
        tf_rep = enc(torch.randn(2, 1, 16000))  # [batch, freq, time]
        id_reim_mask = torch.ones((1, fb_dim, 1))
        assert_allclose(_masks['reim'][0](tf_rep, id_reim_mask, dim=1),
                        tf_rep)


def test_comp_mask(encoder_list):
    for (enc, fb_dim) in encoder_list:
        tf_rep = enc(torch.randn(2, 1, 16000))  # [batch, freq, time]
        id_complex_mask = torch.cat((torch.ones((1, fb_dim // 2, 1)),
                                     torch.zeros((1, fb_dim // 2, 1))),
                                    dim=1)
        assert_allclose(_masks['comp'][0](tf_rep, id_complex_mask, dim=1),
                        tf_rep)
