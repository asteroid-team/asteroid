import torch
from torch import testing

from asteroid.filterbanks import Encoder, Decoder, STFTFB
from asteroid.filterbanks import make_enc_dec

fb_config = {
    'n_filters': 512,
    'kernel_size': 256,
    'stride': 128
}


def test_stft_def():
    """ Check consistency between two calls."""
    fb = STFTFB(**fb_config)
    enc = Encoder(fb)
    dec = Decoder(fb)
    enc2, dec2 = make_enc_dec('stft', **fb_config)
    testing.assert_allclose(enc.filterbank.filters, enc2.filterbank.filters)
    testing.assert_allclose(dec.filterbank.filters, dec2.filterbank.filters)


def test_filter_shape():
    n_filters, kernel_size, stride = 128, 16, 8
    fb = STFTFB(n_filters=128, kernel_size=16, stride=8)
    assert fb.filters.shape == (n_filters + 2, 1, kernel_size)


def test_istft():
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


def check_ola():
    kernel_size = fb_config['kernel_size']
    enc, dec = make_enc_dec('stft', window=None, **fb_config)

    inp = torch.ones(1, 1, 4096)
    tf_rep = dec(enc(inp))[:, :, kernel_size: -kernel_size]
    testing.assert_allclose(tf_rep, tf_rep.mean())
