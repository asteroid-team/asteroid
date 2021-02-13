from unittest import mock
import numpy as np
import pytest
from asteroid.metrics import get_metrics, MetricTracker


@pytest.mark.parametrize("fs", [8000, 16000])
def test_get_metrics(fs):
    mix = np.random.randn(1, 16000)
    clean = np.random.randn(2, 16000)
    est = np.random.randn(2, 16000)
    metrics_dict = get_metrics(mix, clean, est, sample_rate=fs, metrics_list="si_sdr")
    # Test no average & squeezing
    metrics_dict_bis = get_metrics(
        mix[0], clean, est, sample_rate=fs, metrics_list="si_sdr", average=False
    )
    assert float(np.mean(metrics_dict_bis["si_sdr"])) == metrics_dict["si_sdr"]
    assert float(np.mean(metrics_dict_bis["input_si_sdr"])) == metrics_dict["input_si_sdr"]


def test_all_metrics():
    # This is separated because very slow (sdr, pesq, stoi)
    mix = np.random.randn(1, 4000)
    clean = np.random.randn(1, 4000)
    est = np.random.randn(1, 4000)
    get_metrics(mix, clean, est, sample_rate=8000, metrics_list="all")


def test_get_metrics_multichannel():
    mix = np.random.randn(2, 16000)
    clean = np.random.randn(2, 16000)
    est = np.random.randn(2, 16000)
    get_metrics(mix, clean, est, sample_rate=8000, metrics_list="si_sdr", average=False)


@pytest.mark.parametrize("filename", [None, "example.wav"])
def test_error_msg(filename):
    mix = np.random.randn(1, 4000)
    clean = np.random.randn(1, 4000)
    est = np.random.randn(1, 4000)
    expected_msg = f".+si_sdr.+{filename or '<unknown file>'}"
    with mock.patch(
        "pb_bss_eval.evaluation.si_sdr", side_effect=RuntimeError("Fatal error")
    ), pytest.raises(RuntimeError, match=expected_msg):
        metrics_dict = get_metrics(
            mix, clean, est, sample_rate=8000, metrics_list=["si_sdr", "pesq"], filename=filename
        )


@pytest.mark.parametrize("average", [True, False])
@pytest.mark.parametrize("filename", [None, "example.wav"])
def test_ignore_errors(filename, average):
    mix = np.random.randn(1, 4000)
    clean = np.random.randn(1, 4000)
    est = np.random.randn(1, 4000)
    expected_msg = f".+si_sdr.+{filename or '<unknown file>'}.+Fatal error"
    with mock.patch(
        "pb_bss_eval.evaluation.si_sdr", side_effect=RuntimeError("Fatal error")
    ), pytest.warns(RuntimeWarning, match=expected_msg):
        metrics_dict = get_metrics(
            mix,
            clean,
            est,
            sample_rate=8000,
            metrics_list=["si_sdr", "pesq"],
            ignore_metrics_errors=True,
            average=average,
            filename=filename,
        )
    assert metrics_dict["si_sdr"] is None
    assert metrics_dict["pesq"] is not None


def test_metric_tracker():
    metric_tracker = MetricTracker(sample_rate=8000, metrics_list=["si_sdr", "stoi"])
    for i in range(5):
        mix = np.random.randn(1, 4000)
        clean = np.random.randn(1, 4000)
        est = np.random.randn(1, 4000)
        metric_tracker(mix=mix, clean=clean, estimate=est, mix_path=f"path{i}")

    # Test dump & final report
    metric_tracker.final_report()
    metric_tracker.final_report(dump_path="final_metrics.json")

    # Check that kwargs are passed.
    assert "mix_path" in metric_tracker.as_df()
