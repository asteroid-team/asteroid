import torch
import pytest
from torch.testing import assert_allclose
import numpy as np
import soundfile as sf
import asteroid
from asteroid import models
from asteroid.models.base_models import BaseModel
from asteroid.models import ConvTasNet, DPRNNTasNet, DPTNet, LSTMTasNet, DeMask
from asteroid.models import SuDORMRFNet, SuDORMRFImprovedNet


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
    assert_allclose(model1(test_input), reconstructed_model(test_input))


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
    assert_allclose(model1(test_input), reconstructed_model(test_input))


@pytest.mark.parametrize("fb", ["free", "stft", "analytic_free", "param_sinc"])
def test_save_and_load_tasnet(fb):
    model1 = LSTMTasNet(n_src=2, hid_size=4, n_layers=1, n_filters=32, dropout=0.0, fb_name=fb,)
    test_input = torch.randn(1, 800)
    model_conf = model1.serialize()

    reconstructed_model = LSTMTasNet.from_pretrained(model_conf)
    assert_allclose(model1(test_input), reconstructed_model(test_input))


def test_sudormrf():
    model = SuDORMRFNet(
        2, bn_chan=10, num_blocks=4, upsampling_depth=2, kernel_size=21, n_filters=12,
    )
    test_input = torch.randn(1, 801)
    model_conf = model.serialize()

    reconstructed_model = SuDORMRFNet.from_pretrained(model_conf)
    assert_allclose(model(test_input), reconstructed_model(test_input))


def test_sudormrf_imp():
    model = SuDORMRFImprovedNet(
        2, bn_chan=10, num_blocks=4, upsampling_depth=2, kernel_size=21, n_filters=12,
    )
    test_input = torch.randn(1, 801)
    model_conf = model.serialize()

    reconstructed_model = SuDORMRFImprovedNet.from_pretrained(model_conf)
    assert_allclose(model(test_input), reconstructed_model(test_input))


def test_dptnet():
    model = DPTNet(2, ff_hid=10, chunk_size=4, n_repeats=2)
    test_input = torch.randn(1, 801)

    model_conf = model.serialize()
    reconstructed_model = DPTNet.from_pretrained(model_conf)
    assert_allclose(model(test_input), reconstructed_model(test_input))


@pytest.mark.parametrize(
    "model", [LSTMTasNet, ConvTasNet, DPRNNTasNet, DPTNet, SuDORMRFImprovedNet, SuDORMRFNet]
)
def test_get(model):
    retrieved = models.get(model.__name__)
    assert retrieved == model


@pytest.mark.parametrize("wrong", ["wrong_string", 12, object()])
def test_get_errors(wrong):
    with pytest.raises(ValueError):
        # Should raise for anything not a Optimizer instance + unknown string
        models.get(wrong)


def test_register():
    class Custom(BaseModel):
        def __init__(self):
            super().__init__()

    models.register_model(Custom)
    cls = models.get("Custom")
    assert cls == Custom

    with pytest.raises(ValueError):
        models.register_model(models.DPRNNTasNet)


def test_show():
    asteroid.show_available_models()


def test_demask():
    model = DeMask()
    test_input = torch.randn(1, 801)

    model_conf = model.serialize()
    reconstructed_model = DeMask.from_pretrained(model_conf)
    assert_allclose(model.separate(test_input), reconstructed_model(test_input))
