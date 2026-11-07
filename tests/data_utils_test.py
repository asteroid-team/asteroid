"""Tests for online mixing with silent and nonzero audio sources."""

import numpy as np
import pytest
import soundfile as sf
import torch


@pytest.fixture
def collate():
    from asteroid.data.utils import online_mixing_collate

    return online_mixing_collate


@pytest.fixture
def fuss_dataset():
    from asteroid.data.fuss_dataset import FUSSDataset

    return FUSSDataset


def check_batch(collate, targets):
    batch = [(row.sum(0).clone(), row.clone()) for row in targets]
    original = [(mix.clone(), source.clone()) for mix, source in batch]
    mix, sources = collate(batch)
    assert sources.shape == targets.shape
    assert mix.shape == (targets.shape[0], targets.shape[2])
    assert sources.dtype == targets.dtype and mix.dtype == targets.dtype
    assert sources.device == targets.device and mix.device == targets.device
    assert torch.isfinite(sources).all() and torch.isfinite(mix).all()
    assert torch.equal(mix, sources.sum(1))
    for (mix_in, source_in), (mix_before, source_before) in zip(batch, original):
        assert torch.equal(mix_in, mix_before)
        assert torch.equal(source_in, source_before)
    return sources


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch_size,n_src", [(1, 1), (1, 3), (3, 1), (3, 3)])
def test_all_silent(collate, dtype, batch_size, n_src):
    targets = torch.zeros(batch_size, n_src, 8, dtype=dtype)
    assert torch.equal(check_batch(collate, targets), targets)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_silent_donors_and_recipients(collate, monkeypatch, dtype):
    targets = torch.tensor([[0, 2], [3, 0], [0, 4]], dtype=dtype)[..., None].repeat(1, 1, 8)
    permutations = [torch.tensor([1, 2, 0]), torch.tensor([2, 1, 0])]
    calls = []

    def randperm(size):
        assert size == 3
        permutation = permutations[len(calls)]
        calls.append(size)
        return permutation

    monkeypatch.setattr(torch, "randperm", randperm)
    actual = check_batch(collate, targets)
    assert len(calls) == 2
    # Includes zero/zero, positive/zero, zero/positive and positive/positive.
    expected = torch.zeros_like(targets)
    expected[0, 1] = 2
    expected[2, 1] = 4
    torch.testing.assert_close(actual, expected)
    assert torch.count_nonzero(actual[:, 0]) == 0
    assert torch.count_nonzero(actual[1, 1]) == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("quiet", [False, True])
def test_positive_energy_unchanged(collate, monkeypatch, dtype, quiet):
    targets = torch.arange(1, 49, dtype=dtype).reshape(3, 2, 8) / 10
    if quiet:
        targets *= 1e-8 if dtype == torch.float32 else 1e-100
    permutation = torch.tensor([2, 0, 1])
    monkeypatch.setattr(torch, "randperm", lambda size: permutation)
    energies = targets.square().sum(-1, keepdim=True)
    assert (energies > 0).all()
    expected = []
    for i in range(targets.shape[1]):
        donor = targets[permutation, i, :]
        expected.append(donor * torch.sqrt(energies[:, i] / donor.square().sum(-1, keepdim=True)))
    actual = check_batch(collate, targets)
    assert torch.equal(actual, torch.stack(expected, dim=1))
    torch.testing.assert_close(actual.square().sum(-1), energies.squeeze(-1), atol=0, rtol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_seeded_shuffle_unchanged(collate, dtype):
    targets = torch.arange(1, 49, dtype=dtype).reshape(3, 2, 8) / 10
    state = torch.get_rng_state()
    try:
        torch.manual_seed(1729)
        expected = []
        energies = targets.square().sum(-1, keepdim=True)
        for i in range(targets.shape[1]):
            donor = targets[torch.randperm(3), i, :]
            expected.append(
                donor * torch.sqrt(energies[:, i] / donor.square().sum(-1, keepdim=True))
            )
        expected_state = torch.get_rng_state()
        torch.manual_seed(1729)
        actual = check_batch(collate, targets)
        assert torch.equal(actual, torch.stack(expected, dim=1))
        assert torch.equal(torch.get_rng_state(), expected_state)
    finally:
        torch.set_rng_state(state)


def test_fuss_absent_foregrounds(collate, fuss_dataset, tmp_path):
    # Short real audio exercises the reader without a dataset download.
    # The dataset does not enforce its nominal ten-second sample length.
    foreground = np.linspace(-0.25, 0.25, 32, dtype=np.float32)
    bg = tmp_path / "background.wav"
    fg = tmp_path / "foreground.wav"
    mix = tmp_path / "mixture.wav"
    sf.write(str(bg), np.zeros(32, dtype=np.float32), 16000, subtype="FLOAT")
    sf.write(str(fg), foreground, 16000, subtype="FLOAT")
    sf.write(str(mix), foreground, 16000, subtype="FLOAT")
    file_list = tmp_path / "examples.tsv"
    # Include a full-source row so pandas does not infer all-empty columns as floats.
    # Collate only the first two rows to exercise one and zero foregrounds.
    file_list.write_text(f"{mix}\t{bg}\t{fg}\n{bg}\t{bg}\n{mix}\t{bg}\t{fg}\t{fg}\t{fg}\n")
    dataset = fuss_dataset(str(file_list))
    batch = [dataset[0], dataset[1]]
    assert len(batch[0]) == 2
    assert torch.count_nonzero(batch[0][1][1:]) == 0
    assert torch.count_nonzero(batch[1][1]) == 0
    before = [(x.clone(), y.clone()) for x, y in batch]
    inputs, sources = collate(batch)
    assert inputs.shape == (2, 32) and sources.shape == (2, 3, 32)
    assert inputs.dtype == torch.float32 and sources.dtype == torch.float32
    assert torch.isfinite(inputs).all() and torch.isfinite(sources).all()
    assert torch.equal(inputs, sources.sum(1))
    assert torch.count_nonzero(sources[:, 1:]) == 0
    for (x, y), (a, b) in zip(batch, before):
        assert torch.equal(x, a) and torch.equal(y, b)
