import torch
import pytest
from torch.testing import assert_allclose
from asteroid.models import ConvTasNet


def test_convtasnet_def():
    conv_tasnet = ConvTasNet(n_src=2, n_repeats=2, n_blocks=3, bn_chan=16,
                             hid_chan=4, skip_chan=8, n_filters=32)


@pytest.mark.parametrize('fb', ['free', 'stft', 'analytic_free', 'param_sinc'])
def test_save_and_load(fb):
    model1 = ConvTasNet(n_src=2, n_repeats=2, n_blocks=2, bn_chan=16,
                        hid_chan=4, skip_chan=8, n_filters=32, fb_name=fb)
    test_input = torch.randn(1, 800)
    model_conf = model1.serialize()

    reconstructed_model = ConvTasNet.from_pretrained(model_conf)
    assert_allclose(model1.separate(test_input),
                    reconstructed_model(test_input))

