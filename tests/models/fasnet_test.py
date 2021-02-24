import torch
import pytest

from asteroid.models.fasnet import FasNetTAC


@pytest.mark.parametrize("samples", [8372])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("n_mics", [1, 2])
@pytest.mark.parametrize("n_src", [1, 2, 3])
@pytest.mark.parametrize("use_tac", [True, False])
@pytest.mark.parametrize("enc_dim", [4])
@pytest.mark.parametrize("feature_dim", [8])
@pytest.mark.parametrize("window", [2])
@pytest.mark.parametrize("context", [3])
def test_fasnet(batch_size, n_mics, samples, n_src, use_tac, enc_dim, feature_dim, window, context):
    mixture = torch.rand((batch_size, n_mics, samples))
    valid_mics = torch.tensor([n_mics for x in range(batch_size)])
    fasnet = FasNetTAC(
        n_src,
        use_tac=use_tac,
        enc_dim=enc_dim,
        feature_dim=feature_dim,
        window_ms=window,
        context_ms=context,
    )
    fasnet(mixture, valid_mics)
