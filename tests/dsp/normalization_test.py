import numpy as np
from asteroid.dsp.normalization import normalize_estimates


def test_normalization():

    mix = (np.random.rand(1600) - 0.5) * 2  # random [-1,1[
    est = (np.random.rand(2, 1600) - 0.5) * 10
    est_normalized = normalize_estimates(est, mix)

    assert np.max(est_normalized) < 1
    assert np.min(est_normalized) >= -1


def test_silent_estimate_stays_silent():
    mix = np.array([0.5, -0.25])
    est = np.array([[0.0, 0.0], [1.0, -0.5]])
    normalized = normalize_estimates(est, mix)
    assert not np.isnan(normalized).any()
    np.testing.assert_allclose(normalized[0], [0.0, 0.0])
    np.testing.assert_allclose(normalized[1], [0.5, -0.25])
