import torch

from asteroid.losses import ActiveOnlyPITSilencePenalty


def test_active_only_pit_returns_scalar_by_default():
    loss_func = ActiveOnlyPITSilencePenalty()
    est = torch.randn(2, 3, 256)
    tgt = torch.randn(2, 3, 256)

    loss = loss_func(est, tgt)

    assert torch.is_tensor(loss)
    assert loss.ndim == 0


def test_active_only_pit_return_components_schema():
    loss_func = ActiveOnlyPITSilencePenalty()
    est = torch.randn(2, 3, 256)
    tgt = torch.randn(2, 3, 256)

    loss, components = loss_func(est, tgt, return_components=True)

    assert torch.is_tensor(loss)
    assert isinstance(components, dict)
    assert set(components.keys()) == {
        "active_pair",
        "active_l1",
        "active_l2",
        "active_rms_floor",
        "active_diversity",
        "active_activity_bce",
        "silent_rms_penalty",
        "n_active_targets",
    }
    for value in components.values():
        assert torch.is_tensor(value)
        assert value.ndim == 0


def test_silence_penalty_length_invariant_rms_db():
    loss_func = ActiveOnlyPITSilencePenalty(silence_metric="rms_db", silence_margin_db=-45.0)
    amp = 1e-2
    # Same RMS, different durations should produce the same dB penalty.
    est_short = torch.full((1, 2, 1000), amp)
    est_long = torch.full((1, 2, 4000), amp)
    tgt_short = torch.zeros_like(est_short)
    tgt_long = torch.zeros_like(est_long)

    _, comp_short = loss_func(est_short, tgt_short, return_components=True)
    _, comp_long = loss_func(est_long, tgt_long, return_components=True)

    assert torch.allclose(comp_short["silent_rms_penalty"], comp_long["silent_rms_penalty"], atol=1e-5)


def test_active_threshold_mode_rms_vs_sum_energy():
    amp = 2e-4
    threshold = 1e-5
    tgt = torch.zeros(1, 2, 1000)
    tgt[0, 0] = amp
    est = torch.zeros_like(tgt)

    loss_rms = ActiveOnlyPITSilencePenalty(
        threshold=threshold,
        active_threshold_mode="rms",
    )
    _, comp_rms = loss_rms(est, tgt, return_components=True)
    assert comp_rms["n_active_targets"].item() == 0.0

    loss_energy = ActiveOnlyPITSilencePenalty(
        threshold=threshold,
        active_threshold_mode="sum_energy",
    )
    _, comp_energy = loss_energy(est, tgt, return_components=True)
    assert comp_energy["n_active_targets"].item() == 1.0


def test_active_diversity_penalty_nonzero_for_duplicate_active_channels():
    tgt = torch.randn(1, 2, 256)
    est = torch.zeros_like(tgt)
    est[0, 0] = tgt[0, 0]
    est[0, 1] = tgt[0, 0]

    loss_func = ActiveOnlyPITSilencePenalty(
        active_diversity_weight=1.0,
        threshold=1e-8,
    )
    _, comps = loss_func(est, tgt, return_components=True)
    assert comps["active_diversity"].item() > 0.0


def test_active_activity_bce_is_finite_when_enabled():
    tgt = torch.zeros(1, 3, 256)
    tgt[0, 0] = torch.randn(256)
    est = torch.randn(1, 3, 256)

    loss_func = ActiveOnlyPITSilencePenalty(
        active_activity_bce_weight=0.5,
        threshold=1e-8,
    )
    loss, comps = loss_func(est, tgt, return_components=True)
    assert torch.isfinite(loss)
    assert torch.isfinite(comps["active_activity_bce"])
