from .utils import average_arrays_in_dic
from pb_bss_eval import InputMetrics, OutputMetrics

ALL_METRICS = ["si_sdr", "sdr", "sir", "sar", "stoi", "pesq"]


def get_metrics(
    mix,
    clean,
    estimate,
    sample_rate=16000,
    metrics_list="all",
    average=True,
    compute_permutation=False,
):
    """ Get speech separation/enhancement metrics from mix/clean/estimate.

    Args:
        mix (np.array): 'Shape(D, N)' or 'Shape(N, )'.
        clean (np.array): 'Shape(K_source, N)' or 'Shape(N, )'.
        estimate (np.array): 'Shape(K_target, N)' or 'Shape(N, )'.
        sample_rate (int): sampling rate of the audio clips.
        metrics_list (Union [str, list]): List of metrics to compute.
            Defaults to 'all' (['si_sdr', 'sdr', 'sir', 'sar', 'stoi', 'pesq']).
        average (bool): Return dict([float]) if True, else dict([array]).
        compute_permutation (bool): Whether to compute the permutation on
            estimate sources for the output metrics (default False)

    Returns:
        dict: Dictionary with all requested metrics, with `'input_'` prefix
            for metrics at the input (mixture against clean), no prefix at the
            output (estimate against clean). Output format depends on average.

    Examples:
        >>> import numpy as np
        >>> import pprint
        >>> from asteroid.metrics import get_metrics
        >>> mix = np.random.randn(1, 16000)
        >>> clean = np.random.randn(2, 16000)
        >>> est = np.random.randn(2, 16000)
        >>> metrics_dict = get_metrics(mix, clean, est, sample_rate=8000,
        >>>                            metrics_list='all')
        >>> pprint.pprint(metrics_dict)
        {'input_pesq': 1.924380898475647,
         'input_sar': -11.67667585294225,
         'input_sdr': -14.88667106190552,
         'input_si_sdr': -52.43849784881705,
         'input_sir': -0.10419427290163795,
         'input_stoi': 0.015112115177091223,
         'pesq': 1.7713886499404907,
         'sar': -11.610963379923195,
         'sdr': -14.527246041125844,
         'si_sdr': -46.26557128489802,
         'sir': 0.4799929272243427,
         'stoi': 0.022023073540350643}

    """
    if metrics_list == "all":
        metrics_list = ALL_METRICS
    if isinstance(metrics_list, str):
        metrics_list = [metrics_list]
    # For each utterance, we get a dictionary with the input and output metrics
    input_metrics = InputMetrics(
        observation=mix, speech_source=clean, enable_si_sdr=True, sample_rate=sample_rate
    )
    utt_metrics = {"input_" + n: input_metrics[n] for n in metrics_list}

    output_metrics = OutputMetrics(
        speech_prediction=estimate,
        speech_source=clean,
        enable_si_sdr=True,
        sample_rate=sample_rate,
        compute_permutation=compute_permutation,
    )
    utt_metrics.update(output_metrics[metrics_list])
    if average is True:
        return average_arrays_in_dic(utt_metrics)
    else:
        return utt_metrics
