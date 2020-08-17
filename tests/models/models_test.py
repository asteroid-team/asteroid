import torch
import pytest
from torch.testing import assert_allclose
import numpy as np
import soundfile as sf
from asteroid.models import ConvTasNet, DPRNNTasNet, DPTNet
from asteroid.models import SuDORMRF, SuDORMRFImproved


def test_convtasnet_sep():
    nnet = ConvTasNet(
        n_src=2, n_repeats=2, n_blocks=3, bn_chan=16, hid_chan=4, skip_chan=8, n_filters=32
    )
    # Test torch input
    wav = torch.rand(1, 800)
    out = nnet.separate(wav)
    assert isinstance(out, torch.Tensor)
    # Test numpy input
    wav = np.random.randn(1, 800).astype("float32")
    out = nnet.separate(wav)
    assert isinstance(out, np.ndarray)
    # Test str input
    sf.write("tmp.wav", wav[0], 8000)
    nnet.separate("tmp.wav")


@pytest.mark.parametrize("fb", ["free", "stft", "analytic_free", "param_sinc"])
def test_save_and_load_convtasnet(fb):
    model1 = ConvTasNet(
        n_src=2,
        n_repeats=2,
        n_blocks=2,
        bn_chan=16,
        hid_chan=4,
        skip_chan=8,
        n_filters=32,
        fb_name=fb,
    )
    test_input = torch.randn(1, 800)
    model_conf = model1.serialize()

    reconstructed_model = ConvTasNet.from_pretrained(model_conf)
    assert_allclose(model1.separate(test_input), reconstructed_model(test_input))


def test_dprnntasnet_sep():
    nnet = DPRNNTasNet(n_src=2, n_repeats=2, bn_chan=16, hid_size=4, chunk_size=20, n_filters=32)
    # Test torch input
    wav = torch.rand(1, 800)
    out = nnet.separate(wav)
    assert isinstance(out, torch.Tensor)
    # Test numpy input
    wav = np.random.randn(1, 800).astype("float32")
    out = nnet.separate(wav)
    assert isinstance(out, np.ndarray)


@pytest.mark.parametrize("fb", ["free", "stft", "analytic_free", "param_sinc"])
def test_save_and_load_dprnn(fb):
    model1 = DPRNNTasNet(
        n_src=2, n_repeats=2, bn_chan=16, hid_size=4, chunk_size=20, n_filters=32, fb_name=fb
    )
    test_input = torch.randn(1, 800)
    model_conf = model1.serialize()

    reconstructed_model = DPRNNTasNet.from_pretrained(model_conf)
    assert_allclose(model1.separate(test_input), reconstructed_model(test_input))


def test_sudormrf():
    model = SuDORMRF(
        2, out_chan=10, in_chan=10, num_blocks=4, upsampling_depth=2, kernel_size=21, n_filters=10,
    )
    test_input = torch.randn(1, 801)
    model(test_input)


def test_sudormrf_imp():
    model = SuDORMRFImproved(
        2, out_chan=10, in_chan=10, num_blocks=4, upsampling_depth=2, n_filters=10, kernel_size=21,
    )
    test_input = torch.randn(1, 801)
    model(test_input)


def test_dptnet():
    model = DPTNet(2, ff_hid=10, chunk_size=4, n_repeats=2)
    test_input = torch.randn(1, 801)

    model_conf = model.serialize()
    reconstructed_model = DPTNet.from_pretrained(model_conf)
    assert_allclose(model.separate(test_input), reconstructed_model(test_input))
