import torch
import pytest
from torch.testing import assert_allclose
from asteroid.models import (
    DCCRNet,
    DCUNet,
    DeMask,
    ConvTasNet,
    DPRNNTasNet,
    DPTNet,
    LSTMTasNet,
    SuDORMRFNet,
    SuDORMRFImprovedNet,
)


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
            "n_heads": 2,
            "ff_hid": 4,
            "chunk_size": 4,
            "n_repeats": 1,
            "n_filters": 32,
            "kernel_size": 32,
            "stride": 16,
        },
        DCCRNet.__name__: {
            "architecture": "DCCRN-CL",
        },
        DCUNet.__name__: {
            "architecture": "DCUNet-10",
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
        LSTMTasNet.__name__: {
            "n_src": 2,
            "hid_size": 4,
            "n_layers": 1,
            "dropout": 0.0,
            "n_filters": 32,
            "kernel_size": 32,
            "stride": 16,
        },
        SuDORMRFNet.__name__: {
            "n_src": 2,
            "bn_chan": 10,
            "num_blocks": 2,
            "upsampling_depth": 2,
            "n_filters": 32,
            "kernel_size": 21,
            "stride": 10,
        },
        SuDORMRFImprovedNet.__name__: {
            "n_src": 2,
            "bn_chan": 10,
            "num_blocks": 2,
            "upsampling_depth": 2,
            "n_filters": 32,
            "kernel_size": 21,
            "stride": 10,
        },
    }

    return params


@pytest.mark.parametrize(
    "model_def",
    (
        DCCRNet,
        DCUNet,
        DeMask,
    ),
)
@pytest.mark.parametrize(
    "test_data",
    (
        (torch.rand(4740) - 0.5) * 2,
        (torch.rand(1, 4720) - 0.5) * 2,
        (torch.rand(4, 4756) - 0.5) * 2,
        (torch.rand(1, 1, 4801) - 0.5) * 2,
        (torch.rand(3, 1, 4790) - 0.5) * 2,
    ),
)
def test_enhancement_model(small_model_params, model_def, test_data):
    device = get_default_device()
    params = small_model_params[model_def.__name__]
    model = model_def(**params)
    model = model.eval().to(device)
    # Random input uniformly distributed in [-1, 1]
    inputs = ((torch.rand(1, 4701, device=device) - 0.5) * 2,)
    traced = torch.jit.trace(model, inputs)

    # check forward
    test_data = test_data.to(device)
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
        SuDORMRFNet,
        SuDORMRFImprovedNet,
    ),
)
@pytest.mark.parametrize(
    "test_data",
    (
        (torch.rand(240) - 0.5) * 2,
        (torch.rand(1, 220) - 0.5) * 2,
        (torch.rand(3, 250) - 0.5) * 2,
        (torch.rand(1, 1, 301) - 0.5) * 2,
    ),
)
def test_trace_bss_model(small_model_params, model_def, test_data):
    device = get_default_device()
    params = small_model_params[model_def.__name__]
    model = model_def(**params)
    model = model.eval().to(device)
    # Random input uniformly distributed in [-1, 1]
    inputs = ((torch.rand(1, 201, device=device) - 0.5) * 2,)
    traced = torch.jit.trace(model, inputs)

    # check forward
    test_data = test_data.to(device)
    with torch.no_grad():
        ref = model(test_data)
        out = traced(test_data)
        assert_allclose(ref, out)


def get_default_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# def test_padder():
#     model_def = SuDORMRFNet
#     params = small_model_params[model_def.__name__]
#     model = model_def(**params)


if __name__ == "__main__":
    model_def = SuDORMRFNet
    # params = small_model_params[model_def.__name__]
    params = {
        "stride": 10,
        "n_src": 2,
        "bn_chan": 10,
        "num_blocks": 2,
        "upsampling_depth": 2,
        "n_filters": 32,
        "kernel_size": 21,
    }
    model = model_def(**params)
    u = torch.randn(3, 250)
    import ipdb

    ipdb.set_trace()
    v = model(u)

    traced = torch.jit.trace(model, torch.randn(1, 210))
    hey = traced(u)

    assert_allclose(v, hey)
