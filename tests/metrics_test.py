import numpy as np
import pytest
from asteroid.metrics import get_metrics


@pytest.mark.parametrize("fs", [8000, 16000])
def test_get_metrics(fs):
    mix = np.random.randn(1, 16000)
    clean = np.random.randn(2, 16000)
    est = np.random.randn(2, 16000)
    metrics_dict = get_metrics(mix, clean, est, sample_rate=fs,
                               metrics_list='si_sdr')
    # Test no average & squeezing
    metrics_dict_bis = get_metrics(mix[0], clean, est, sample_rate=fs,
                                   metrics_list='si_sdr', average=False)
    assert float(np.mean(metrics_dict_bis['si_sdr'])) == metrics_dict['si_sdr']
    assert (float(np.mean(metrics_dict_bis['input_si_sdr'])) ==
            metrics_dict['input_si_sdr'])


def test_all_metrics():
    # This is separated because very slow (sdr, pesq, stoi)
    mix = np.random.randn(1, 4000)
    clean = np.random.randn(1, 4000)
    est = np.random.randn(1, 4000)
    metrics_dict = get_metrics(mix, clean, est, sample_rate=8000,
                               metrics_list='all')


def test_get_metrics_multichannel():
    mix = np.random.randn(2, 16000)
    clean = np.random.randn(2, 16000)
    est = np.random.randn(2, 16000)
    metrics_dict_bis = get_metrics(mix, clean, est, sample_rate=8000,
                               metrics_list='si_sdr', average=False)