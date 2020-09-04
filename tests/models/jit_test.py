import torch
import pytest
from torch.testing import assert_allclose
from asteroid.models import DeMask, ConvTasNet, DPRNNTasNet, DPTNet, LSTMTasNet
from torch.testing._internal.jit_utils import JitTestCase


@pytest.fixture(scope='module')
def small_model_params():
    params = {
        ConvTasNet.__name__: {
            'n_src': 2,
            'n_repeats': 2,
            'n_blocks': 3,
            'bn_chan': 16,
            'hid_chan': 4,
            'skip_chan': 8,
            'n_filters': 32,
        },
        DPRNNTasNet.__name__: {
            'n_src': 2,
            'n_repeats': 2,
            'bn_chan': 16,
            'hid_size': 4,
            'chunk_size': 20,
            'n_filters': 32,
        },
        DPTNet.__name__: {
            'n_src': 2,
            'ff_hid': 10,
            'chunk_size': 4,
            'n_repeats': 2,
        },
    }

    return params


@pytest.mark.parametrize('model_def',
    (
        ConvTasNet,
        DPRNNTasNet,
        DPTNet,
    )
)
def test_trace_bss_model(small_model_params, model_def):
    filter_bank_types = ['free', 'stft', 'analytic_free', 'param_sinc']
    device = get_default_device()
    # Random input uniformly distributed in [-1, 1]
    inputs = ((torch.rand(1, 1000, device=device) - 0.5) * 2,)
    test_data = torch.rand(1, 32, device=device)
    for filter_bank in filter_bank_types:
        params = small_model_params[model_def.__name__]
        model = model_def(**params, fb_name=filter_bank)
        model = model.eval().to(device)
        traced = torch.jit.trace(model, inputs)

        # check forward
        with torch.no_grad():
            ref = model(test_data)
            out = traced(test_data)
            assert torch.allclose(ref, out)


def get_default_device():
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'
