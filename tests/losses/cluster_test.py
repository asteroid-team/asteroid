import pytest
import torch

from asteroid.losses.cluster import deep_clustering_loss


def _affinity_reference(embedding, labels, mask=None):
    labels = labels.reshape(labels.shape[0], -1)
    if mask is None:
        weights = torch.ones_like(labels, dtype=embedding.dtype)
    else:
        weights = mask.reshape_as(labels).to(embedding)
    affinity = (labels.unsqueeze(-1) == labels.unsqueeze(-2)).to(embedding)
    error = embedding @ embedding.transpose(-1, -2) - affinity
    error = error * weights.unsqueeze(-1) * weights.unsqueeze(-2)
    return error.square().sum((-1, -2)) / weights.sum(-1)


@pytest.mark.parametrize("source_ids", [(1,), (0, 2), (1, 2), (0, 1, 2)])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("mask_kind", ["none", "bool", "float"])
def test_dc_missing_source_ids(source_ids, batch_size, mask_kind):
    labels = torch.tensor(source_ids)[torch.arange(6) % len(source_ids)]
    labels = labels.reshape(1, 2, 3).repeat(batch_size, 1, 1)
    generator = torch.Generator().manual_seed(42)
    embedding = torch.nn.functional.normalize(
        torch.randn(batch_size, 6, 3, generator=generator), dim=-1
    ).requires_grad_()
    if mask_kind == "none":
        mask = None
    elif mask_kind == "bool":
        mask = torch.tensor([True, False, True, True, False, True]).reshape(1, 2, 3)
    else:
        mask = torch.tensor([0.25, 1.0, 0.5, 0.75, 0.0, 1.0]).reshape(1, 2, 3)
    if mask is not None:
        mask = mask.repeat(batch_size, 1, 1)
    labels_before, embedding_before = labels.clone(), embedding.detach().clone()
    mask_before = None if mask is None else mask.clone()

    actual = deep_clustering_loss(embedding, labels, mask)
    expected = _affinity_reference(embedding, labels, mask)
    torch.testing.assert_close(actual, expected)
    (actual_grad,) = torch.autograd.grad(actual.sum(), embedding, retain_graph=True)
    (expected_grad,) = torch.autograd.grad(expected.sum(), embedding)
    torch.testing.assert_close(actual_grad, expected_grad)
    assert torch.equal(labels, labels_before)
    assert torch.equal(embedding.detach(), embedding_before)
    assert mask is None or torch.equal(mask, mask_before)


def test_dc_source_renaming_and_batch_composition():
    generator = torch.Generator().manual_seed(12)
    embedding = torch.randn(1, 6, 3, generator=generator).requires_grad_()
    zeros = torch.zeros(1, 2, 3, dtype=torch.long)
    ones = torch.ones_like(zeros)
    first = deep_clustering_loss(embedding, zeros)
    renamed = deep_clustering_loss(embedding, ones)
    batched = deep_clustering_loss(embedding.repeat(2, 1, 1), torch.cat([ones, zeros]))
    torch.testing.assert_close(first, renamed)
    torch.testing.assert_close(renamed, batched[:1])
    (first_grad,) = torch.autograd.grad(first.sum(), embedding, retain_graph=True)
    (renamed_grad,) = torch.autograd.grad(renamed.sum(), embedding)
    torch.testing.assert_close(first_grad, renamed_grad)


def test_dc_argmax_targets_and_embedding_head_gradients():
    generator = torch.Generator().manual_seed(24)
    waveform = torch.randn(32, generator=generator)
    sources = torch.stack([0.25 * waveform, waveform])
    magnitudes = (
        torch.stft(sources, n_fft=8, hop_length=4, window=torch.hann_window(8), return_complex=True)
        .abs()
        .unsqueeze(0)
    )
    real_mask = magnitudes / magnitudes.sum(1, keepdim=True).clamp_min(1e-8)
    labels = real_mask.argmax(1)
    assert labels.unique().tolist() == [1]
    features = torch.randn(1, labels.numel(), 4, generator=generator)
    head = torch.nn.Linear(4, 3)
    with torch.no_grad():
        head.weight.copy_(torch.linspace(-0.3, 0.4, 12).reshape(3, 4))
        head.bias.copy_(torch.tensor([-0.1, 0.0, 0.1]))
    embedding = torch.nn.functional.normalize(head(features).tanh(), dim=-1)

    actual = deep_clustering_loss(embedding, labels)
    expected = _affinity_reference(embedding, labels)
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual.sum(), tuple(head.parameters()), retain_graph=True)
    expected_grad = torch.autograd.grad(expected.sum(), tuple(head.parameters()))
    for actual_value, expected_value in zip(actual_grad, expected_grad):
        assert torch.isfinite(actual_value).all()
        torch.testing.assert_close(actual_value, expected_value)
