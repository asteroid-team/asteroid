import numpy as np
from asteroid.dsp.normalization import normalize_estimates


def test_normalization():

    mix = (np.random.rand(1600) - 0.5) * 2  # random [-1,1[
    est = (np.random.rand(2, 1600) - 0.5) * 10
    est_normalized = normalize_estimates(est, mix)

    assert np.max(est_normalized) < 1
    assert np.min(est_normalized) >= -1
