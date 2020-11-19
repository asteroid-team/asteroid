import pytest
import torch
from torch import testing
import numpy as np
from scipy.signal import get_window
import random

from asteroid.filterbanks import Encoder, Decoder, STFTFB
from asteroid.filterbanks import make_enc_dec, griffin_lim, misi
from asteroid.filterbanks.stft_fb import perfect_synthesis_window
from asteroid.filterbanks import transforms
from asteroid.filterbanks.melgram_fb import MelScale, MelGramFB


def fb_config_list():
    keys = ["n_filters", "kernel_size", "stride"]
    param_list = [
        [256, 256, 128],  # Usual STFT, 50% overlap
        [256, 256, 64],  # Usual STFT, 25% overlap
        [512, 32, 16],  # Overcomplete STFT, 50% overlap
    ]
    return [dict(zip(keys, values)) for values in param_list]


@pytest.mark.parametrize("fb_config", fb_config_list())
def test_stft_def(fb_config):
    """ Check consistency between two calls."""
    fb = STFTFB(**fb_config)
    enc = Encoder(fb)
    dec = Decoder(fb)
    enc2, dec2 = make_enc_dec("stft", **fb_config)
    testing.assert_allclose(enc.filterbank.filters(), enc2.filterbank.filters())
    testing.assert_allclose(dec.filterbank.filters(), dec2.filterbank.filters())


@pytest.mark.parametrize("n_filters", (13, 257))
def test_stft_def_error(n_filters):
    with pytest.raises(ValueError) as err:
        STFTFB(n_filters, n_filters)
    assert str(err.value) == f"n_filters must be even, got {n_filters}"


@pytest.mark.parametrize("fb_config", fb_config_list())
def test_stft_windows(fb_config):
    kernel_size = fb_config["kernel_size"]
    win = np.hanning(kernel_size)
    STFTFB(**fb_config, window=win)
    with pytest.raises(AssertionError):
        win = np.hanning(kernel_size + 1)
        STFTFB(**fb_config, window=win)


@pytest.mark.parametrize("fb_config", fb_config_list())
def test_filter_shape(fb_config):
    # Instantiate STFT
    fb = STFTFB(**fb_config)
    # Check filter shape.
    assert fb.filters().shape == (fb_config["n_filters"] + 2, 1, fb_config["kernel_size"])


@pytest.mark.parametrize("fb_config", fb_config_list())
def test_perfect_istft_default_parameters(fb_config):
    """ Unit test perfect reconstruction with default values. """
    kernel_size = fb_config["kernel_size"]
    enc, dec = make_enc_dec("stft", **fb_config)
    inp_wav = torch.randn(2, 1, 32000)
    out_wav = dec(enc(inp_wav))[:, :, kernel_size:-kernel_size]
    inp_test = inp_wav[:, :, kernel_size:-kernel_size]
    testing.assert_allclose(inp_test, out_wav)


@pytest.mark.parametrize("fb_config", fb_config_list())
@pytest.mark.parametrize(
    "analysis_window_name", ["blackman", "hamming", "hann", "bartlett", "boxcar"]
)
def test_perfect_resyn_window(fb_config, analysis_window_name):
    """ Unit test perfect reconstruction """
    kernel_size = fb_config["kernel_size"]
    window = get_window(analysis_window_name, kernel_size)

    enc = Encoder(STFTFB(**fb_config, window=window))
    # Compute window for perfect resynthesis
    synthesis_window = perfect_synthesis_window(enc.filterbank.window, enc.stride)
    dec = Decoder(STFTFB(**fb_config, window=synthesis_window))
    inp_wav = torch.ones(1, 1, 32000)
    out_wav = dec(enc(inp_wav))[:, :, kernel_size:-kernel_size]
    inp_test = inp_wav[:, :, kernel_size:-kernel_size]
    testing.assert_allclose(inp_test, out_wav)


@pytest.mark.parametrize("fb_config", fb_config_list())
@pytest.mark.parametrize("feed_istft", [True, False])
@pytest.mark.parametrize("feed_angle", [True, False])
def test_griffinlim(fb_config, feed_istft, feed_angle):
    stft = Encoder(STFTFB(**fb_config))
    istft = None if not feed_istft else Decoder(STFTFB(**fb_config))
    wav = torch.randn(2, 1, 8000)
    spec = stft(wav)
    tf_mask = torch.sigmoid(torch.randn_like(spec))
    masked_spec = spec * tf_mask
    mag = transforms.take_mag(masked_spec, -2)
    angles = None if not feed_angle else transforms.angle(masked_spec, -2)
    griffin_lim(mag, stft, angles=angles, istft_dec=istft, n_iter=3)


@pytest.mark.parametrize("fb_config", fb_config_list())
@pytest.mark.parametrize("feed_istft", [True, False])
@pytest.mark.parametrize("feed_angle", [True, False])
def test_misi(fb_config, feed_istft, feed_angle):
    stft = Encoder(STFTFB(**fb_config))
    istft = None if not feed_istft else Decoder(STFTFB(**fb_config))
    n_src = 3
    # Create mixture
    wav = torch.randn(2, 1, 8000)
    spec = stft(wav).unsqueeze(1)
    # Create n_src masks on mixture spec and apply them
    shape = list(spec.shape)
    shape[1] *= n_src
    tf_mask = torch.sigmoid(torch.randn(*shape))
    masked_specs = spec * tf_mask
    # Separate mag and angle.
    mag = transforms.take_mag(masked_specs, -2)
    angles = None if not feed_angle else transforms.angle(masked_specs, -2)
    est_wavs = misi(wav, mag, stft, angles=angles, istft_dec=istft, n_iter=2)
    # We actually don't know the last dim because ISTFT(STFT()) cuts the end
    assert est_wavs.shape[:-1] == (2, n_src)


@pytest.mark.parametrize("n_filters", [64, 30])
@pytest.mark.parametrize("n_mels", [10, 40, None])
@pytest.mark.parametrize("norm", ["slaney", None])
@pytest.mark.parametrize("ndim", [3, 4, 5])
def test_melscale(n_filters, n_mels, norm, ndim):
    n_mels = n_mels if n_mels is not None else n_filters // 2 + 1
    mel_scale = MelScale(n_filters=n_filters, n_mels=n_mels, norm=norm)
    tensor_shape = tuple([random.randint(1, 3) for _ in range(ndim - 2)]) + (n_filters + 2, 10)
    spec = torch.randn(tensor_shape)
    mel_spec = mel_scale(spec)
    assert spec.shape[:-2] == mel_spec.shape[:-2]
    assert mel_spec.shape[-2] == n_mels


@pytest.mark.parametrize("n_filters", [64, 30])
@pytest.mark.parametrize("n_mels", [10, 40, None])
@pytest.mark.parametrize("ndim", [3, 4, 5])
def test_melgram_encoder(n_filters, n_mels, ndim):
    n_mels = n_mels if n_mels is not None else n_filters // 2 + 1
    melgram_fb = MelGramFB(n_filters=n_filters, kernel_size=n_filters, n_mels=n_mels)
    enc = Encoder(melgram_fb)
    tensor_shape = tuple([random.randint(2, 3) for _ in range(ndim - 1)]) + (4000,)
    wav = torch.randn(tensor_shape)
    mel_spec = enc(wav)
    assert wav.shape[:-1] == mel_spec.shape[:-2]
    assert mel_spec.shape[-2] == n_mels
    conf = melgram_fb.get_config()
