from typing import Tuple

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


@torch.no_grad()
def assert_consistency(model, traced, tensor):
    ref = model(tensor)
    out = traced(tensor)
    assert_allclose(ref, out)


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
            "stft_kernel_size": 256,
            "architecture": "mini",
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
            "n_layers": 2,
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


@pytest.mark.parametrize("model_def", [DCCRNet, DeMask])
@pytest.mark.parametrize(
    "test_data",
    (
        (torch.rand(2001) - 0.5) * 2,
        (torch.rand(1, 4720) - 0.5) * 2,
        (torch.rand(4, 1100) - 0.5) * 2,
        (torch.rand(1, 1, 1502) - 0.5) * 2,
        (torch.rand(3, 1, 4301) - 0.5) * 2,
    ),
)
def test_enhancement_model(small_model_params, model_def, test_data):
    device = get_default_device()
    params = small_model_params[model_def.__name__]
    model = model_def(**params)
    model = model.eval().to(device)
    # Random input uniformly distributed in [-1, 1]
    inputs = ((torch.rand(1, 2500, device=device) - 0.5) * 2,)
    traced = torch.jit.trace(model, inputs)

    assert_consistency(model=model, traced=traced, tensor=test_data.to(device))


@pytest.mark.parametrize("test_shape", [(2,), (3, 1)])
@pytest.mark.parametrize("matching_samples", [4701, 8800, 17001])
def test_dcunet_model(test_shape: Tuple, matching_samples):
    n_samples = 5010
    device = get_default_device()
    model = DCUNet(architecture="mini", fix_length_mode="pad").eval().to(device)
    # Random input uniformly distributed in [-1, 1]
    inputs = torch.rand(1, n_samples, device=device)
    traced = torch.jit.trace(model, (inputs,))

    test_data = torch.rand(*test_shape, matching_samples, device=device)
    assert_consistency(model=model, traced=traced, tensor=test_data.to(device))


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
        (torch.rand(2, 1, 501) - 0.5) * 2,
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

    assert_consistency(model=model, traced=traced, tensor=test_data.to(device))


def get_default_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
