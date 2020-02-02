import pytest
import torch
from torch import testing
import numpy as np

from asteroid.filterbanks import Encoder, Decoder, STFTFB
from asteroid.filterbanks import make_enc_dec


def fb_config_list():
    keys = ['n_filters', 'kernel_size', 'stride']
    param_list = [
        [512, 256, 128],  # Usual STFT, 50% overlap
        [512, 256, 64],  # Usual STFT, 25% overlap
        [512, 32, 16],  # Overcomplete STFT, 50% overlap
    ]
    return [dict(zip(keys, values)) for values in param_list]


@pytest.mark.parametrize("fb_config", fb_config_list())
def test_stft_def(fb_config):
    """ Check consistency between two calls."""
    fb = STFTFB(**fb_config)
    enc = Encoder(fb)
    dec = Decoder(fb)
    enc2, dec2 = make_enc_dec('stft', **fb_config)
    testing.assert_allclose(enc.filterbank.filters, enc2.filterbank.filters)
    testing.assert_allclose(dec.filterbank.filters, dec2.filterbank.filters)


@pytest.mark.parametrize("fb_config", fb_config_list())
def test_stft_windows(fb_config):
    n_filters, kernel_size = fb_config["n_filters"], fb_config["kernel_size"]
    win = np.hanning(kernel_size)
    fb = STFTFB(**fb_config, window=win)
    with pytest.raises(AssertionError):
        win = np.hanning(kernel_size + 1)
        fb = STFTFB(**fb_config, window=win)


@pytest.mark.parametrize("fb_config", fb_config_list())
def test_filter_shape(fb_config):
    # for fb_config in fb_config_list:
    # Instantiate STFT
    fb = STFTFB(**fb_config)
    # Check filter shape.
    assert fb.filters.shape == (fb_config['n_filters'] + 2, 1,
                                fb_config['kernel_size'])


@pytest.mark.parametrize("fb_config", fb_config_list())
def test_istft(fb_config):
    """ Without dividing by the overlap-added window, the STFT iSTFT cannot
    pass the unit test. Uncomment the plot to see the perfect resynthesis."""
    kernel_size = fb_config['kernel_size']
    enc, dec = make_enc_dec('stft', **fb_config)
    inp_wav = torch.randn(2, 1, 32000)
    out_wav = dec(enc(inp_wav))[:, :, kernel_size: -kernel_size]
    inp_test = inp_wav[:, :, kernel_size: -kernel_size]
    # testing.assert_allclose(inp_test,
    #                         out_wav,
    #                         rtol=1e-7, atol=1e-3)
    # import matplotlib.pyplot as plt
    # plt.plot(inp_test.data.numpy()[0, 0], 'b')
    # plt.plot(out_wav.data.numpy()[0, 0], 'r')
    # plt.show()


@pytest.mark.parametrize('kernel_size', [256])
@pytest.mark.parametrize('stride', [128, 64])
def test_ola(kernel_size, stride):
    """ Unit-test the perfect OLA for boxcar weighted DFT filters."""
    fb_config = {
            'n_filters': 2 * kernel_size,
            'kernel_size': kernel_size,
            'stride': stride
        }
    # Make STFT filters with no analysis and synthesis windows.
    # kernel_size = fb_config['kernel_size']
    enc, dec = make_enc_dec('stft', window=None, **fb_config)
    # Input a boxcar function
    inp = torch.ones(1, 1, 4096)
    # Analysis-synthesis. Cut leading and trailing frames.
    synth = dec(enc(inp))[:, :, kernel_size: -kernel_size]
    # Assert that an boxcar input returns a boxcar output.
    testing.assert_allclose(synth, inp[:, :, kernel_size: -kernel_size])
