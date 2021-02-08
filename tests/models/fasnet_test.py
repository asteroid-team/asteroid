import torch
import pytest

from asteroid.models.fasnet import FasNetTAC


@pytest.mark.parametrize("samples", [8372])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("n_mics", [1, 2])
@pytest.mark.parametrize("n_src", [1, 2, 3])
@pytest.mark.parametrize("use_tac", [True, False])
def test_fasnet(batch_size, n_mics, samples, n_src, use_tac):
    mixture = torch.rand((batch_size, n_mics, samples))
    valid_mics = torch.tensor([n_mics for x in range(batch_size)])
    fasnet = FasNetTAC(n_src, use_tac=use_tac, enc_dim=8)
    fasnet(mixture, valid_mics)
