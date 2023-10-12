import torch
import pytest
from torch.testing import assert_close
from asteroid_filterbanks import make_enc_dec
from asteroid.models.base_models import BaseEncoderMaskerDecoder


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
    model = DummyModel(fb_name=filter_bank_name)
    model = model.eval()

    inputs = ((torch.rand(1, 200) - 0.5) * 2,)
    traced = torch.jit.trace(model, inputs)
    with torch.no_grad():
        res = model(inference_data)
        out = traced(inference_data)
        assert_close(res, out)


class DummyModel(BaseEncoderMaskerDecoder):
    def __init__(
        self,
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
        masker = torch.nn.Identity()
        super().__init__(encoder, masker, decoder, encoder_activation=encoder_activation)
