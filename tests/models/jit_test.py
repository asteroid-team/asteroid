import torch
import pytest
from torch.testing import assert_allclose
from asteroid.filterbanks import make_enc_dec
from asteroid.models import DeMask, ConvTasNet, DPRNNTasNet, DPTNet, LSTMTasNet
from asteroid.models.base_models import BaseEncoderMaskerDecoder


@pytest.fixture(scope="module")
def small_model_params():
    params = {
        ConvTasNet.__name__: {
            "n_src": 2,
            "n_repeats": 2,
            "n_blocks": 2,
            "bn_chan": 8,
            "hid_chan": 4,
            "skip_chan": 4,
            "n_filters": 32,
            "kernel_size": 32,
            "stride": 16,
        },
        DPRNNTasNet.__name__: {
            "n_src": 2,
            "n_repeats": 2,
            "bn_chan": 8,
            "hid_size": 4,
            "chunk_size": 3,
            "n_filters": 32,
            "kernel_size": 32,
            "stride": 16,
        },
        DPTNet.__name__: {
            "n_src": 2,
            "ff_hid": 4,
            "chunk_size": 3,
            "n_repeats": 1,
            "n_filters": 32,
            "kernel_size": 32,
            "stride": 16,
        },
        LSTMTasNet.__name__: {
            "n_src": 2,
            "hid_size": 4,
            "n_layers": 1,
            "dropout": 0.0,
            "n_filters": 32,
            "kernel_size": 32,
            "stride": 16,
        },
        DeMask.__name__: {
            "input_type": "mag",
            "output_type": "mag",
            "hidden_dims": [64],
            "dropout": 0,
            "activation": "relu",
            "mask_act": "relu",
            "norm_type": "gLN",
            "stride": 16,
            "n_filters": 32,
            "kernel_size": 32,
        },
    }

    return params


def test_enhancement_model(small_model_params):
    params = small_model_params["DeMask"]
    filter_banks = ["free"]  # , "stft", "analytic_free", "param_sinc"]
    device = get_default_device()
    inputs = ((torch.rand(1, 200, device=device) - 0.5) * 2,)
    test_data = torch.rand(1, 220, device=device)
    for filter_bank in filter_banks:
        model = DeMask(**params, fb_type=filter_bank).eval().to(device)
        traced = torch.jit.trace(model, inputs)

        # check forward
        with torch.no_grad():
            ref = model(test_data)
            out = traced(test_data)
            assert_allclose(ref, out)


@pytest.mark.parametrize(
    "model_def",
    (
        ConvTasNet,
        DPRNNTasNet,
        DPTNet,
        LSTMTasNet,
    ),
)
def test_trace_bss_model(small_model_params, model_def):
    filter_bank_types = ["free"]  # , "stft", "analytic_free", "param_sinc"]
    device = get_default_device()
    # Random input uniformly distributed in [-1, 1]
    inputs = ((torch.rand(1, 200, device=device) - 0.5) * 2,)
    test_data = torch.rand(1, 220, device=device)
    for filter_bank in filter_bank_types:
        params = small_model_params[model_def.__name__]
        model = model_def(**params, fb_name=filter_bank)
        model = model.eval().to(device)
        traced = torch.jit.trace(model, inputs)

        # check forward
        with torch.no_grad():
            ref = model(test_data)
            out = traced(test_data)
            assert_allclose(ref, out)


@pytest.mark.parametrize(
    "filter_bank_name",
    ("free", "stft", "analytic_free", "param_sinc"),
)
@pytest.mark.parametrize(
    "inference_data",
    (
        (torch.rand(240) - 0.5) * 2,
        (torch.rand(1, 220) - 0.5) * 2,
        (torch.rand(4, 256) - 0.5) * 2,
        (torch.rand(1, 3, 312) - 0.5) * 2,
        (torch.rand(3, 2, 128) - 0.5) * 2,
        (torch.rand(1, 1, 3, 212) - 0.5) * 2,
        (torch.rand(2, 4, 3, 128) - 0.5) * 2,
    ),
)
def test_jit_filterbanks(filter_bank_name, inference_data):
    device = get_default_device()
    model = DummyModel(fb_name=filter_bank_name)
    model = model.eval()

    inputs = ((torch.rand(1, 1000) - 0.5) * 2,)
    traced = torch.jit.trace(model, inputs)
    with torch.no_grad():
        res = model(inference_data)
        out = traced(inference_data)
        assert_allclose(res, out)


class DummyModel(BaseEncoderMaskerDecoder):
    def __init__(
        self,
        in_chan=None,
        fb_name="free",
        kernel_size=16,
        n_filters=32,
        stride=8,
        encoder_activation=None,
        **fb_kwargs,
    ):
        encoder, decoder = make_enc_dec(
            fb_name, kernel_size=kernel_size, n_filters=n_filters, stride=stride, **fb_kwargs
        )
        n_feats = encoder.n_feats_out
        if in_chan is not None:
            assert in_chan == n_feats, (
                "Number of filterbank output channels"
                " and number of input channels should "
                "be the same. Received "
                f"{n_feats} and {in_chan}"
            )
        # Update in_chan
        masker = torch.nn.Identity()
        super().__init__(encoder, masker, decoder, encoder_activation=encoder_activation)


def get_default_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
