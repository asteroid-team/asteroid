import numpy as np
import pytest
from asteroid.metrics import get_metrics


@pytest.mark.parametrize("fs", [8000, 16000])
def test_get_metrics(fs):
    mix = np.random.randn(1, 16000)
    clean = np.random.randn(2, 16000)
    est = np.random.randn(2, 16000)
    metrics_dict = get_metrics(mix, clean, est, sample_rate=fs,
                               metrics_list='all')
    # Test no average & squeezing
    metrics_dict_bis = get_metrics(mix[0], clean, est, sample_rate=fs,
                                   metrics_list='all', average=False)
    assert float(np.mean(metrics_dict_bis['pesq'])) == metrics_dict["pesq"]
    assert float(np.mean(metrics_dict_bis['sdr'])) == metrics_dict["sdr"]
    assert float(np.mean(metrics_dict_bis['pesq'])) == metrics_dict["pesq"]
    assert float(np.mean(metrics_dict_bis['pesq'])) == metrics_dict["pesq"]
