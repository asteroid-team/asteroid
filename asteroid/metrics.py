import warnings
import traceback
from collections import Counter
from typing import List

import pandas as pd
import numpy as np
from pb_bss_eval import InputMetrics, OutputMetrics

from .utils import average_arrays_in_dic

ALL_METRICS = ["si_sdr", "sdr", "sir", "sar", "stoi", "pesq"]


def get_metrics(
    mix,
    clean,
    estimate,
    sample_rate=16000,
    metrics_list="all",
    average=True,
    compute_permutation=False,
    ignore_metrics_errors=False,
    filename=None,
):
    r"""Get speech separation/enhancement metrics from mix/clean/estimate.

    Args:
        mix (np.array): mixture array.
        clean (np.array): reference array.
        estimate (np.array): estimate array.
        sample_rate (int): sampling rate of the audio clips.
        metrics_list (Union [str, list]): List of metrics to compute.
            Defaults to 'all' (['si_sdr', 'sdr', 'sir', 'sar', 'stoi', 'pesq']).
        average (bool): Return dict([float]) if True, else dict([array]).
        compute_permutation (bool): Whether to compute the permutation on
            estimate sources for the output metrics (default False)
        ignore_metrics_errors (bool): Whether to ignore errors that occur in
            computing the metrics. A warning will be printed instead.
        filename (str, optional): If computing a metric fails, print this
            filename along with the exception/warning message for debugging purposes.

    Shape:
        - mix: :math:`(D, N)` or `(N, )`.
        - clean: :math:`(K\_source, N)` or `(N, )`.
        - estimate: :math:`(K\_target, N)` or `(N, )`.

    Returns:
        dict: Dictionary with all requested metrics, with `'input_'` prefix
        for metrics at the input (mixture against clean), no prefix at the
        output (estimate against clean). Output format depends on average.

    Examples
        >>> import numpy as np
        >>> import pprint
        >>> from asteroid.metrics import get_metrics
        >>> mix = np.random.randn(1, 16000)
        >>> clean = np.random.randn(2, 16000)
        >>> est = np.random.randn(2, 16000)
        >>> metrics_dict = get_metrics(mix, clean, est, sample_rate=8000,
        ...                            metrics_list='all')
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
    output_metrics = OutputMetrics(
        speech_prediction=estimate,
        speech_source=clean,
        enable_si_sdr=True,
        sample_rate=sample_rate,
        compute_permutation=compute_permutation,
    )
    utt_metrics = {}
    for src, prefix in [(input_metrics, "input_"), (output_metrics, "")]:
        for metric in metrics_list:
            # key: eg. "input_pesq" or "pesq"
            key = prefix + metric
            try:
                utt_metrics[key] = src[metric]
            except Exception as err:
                if ignore_metrics_errors:
                    warnings.warn(
                        f"Error computing {key} for {filename or '<unknown file>'},"
                        f" ignoring. Error was: {err}",
                        RuntimeWarning,
                    )
                    traceback.print_stack()
                    utt_metrics[key] = None
                else:
                    raise RuntimeError(
                        f"Error computing {key} for {filename or '<unknown file>'}"
                    ) from err
    if average:
        return average_arrays_in_dic(utt_metrics)
    else:
        return utt_metrics


class MockWERTracker:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return dict()

    def final_report_as_markdown(self):
        return ""


class WERTracker:
    """Word Error Rate Tracker. Subject to change.

    Args:
        model_name (str): Name of the petrained model to use.
        trans_df (dataframe): Containing field `utt_id` and `text`.
            See librimix/ConvTasNet recipe.
    """

    def __init__(self, model_name, trans_df):

        from espnet2.bin.asr_inference import Speech2Text
        from espnet_model_zoo.downloader import ModelDownloader

        self.model_name = model_name
        d = ModelDownloader()
        self.asr_model = Speech2Text(**d.download_and_unpack(model_name))
        self.input_txt_list = []
        self.clean_txt_list = []
        self.output_txt_list = []
        self.sample_rate = int(d.data_frame[d.data_frame["name"] == model_name]["fs"])
        self.trans_df = trans_df
        self.trans_dic = self._df_to_dict(trans_df)
        self.mix_counter = Counter()
        self.clean_counter = Counter()
        self.est_counter = Counter()

    def __call__(
        self,
        *,
        mix: np.ndarray,
        clean: np.ndarray,
        estimate: np.ndarray,
        sample_rate: int,
        wav_id: List[str],
    ):
        """Compute and store best hypothesis for the mixture and the estimates"""
        if sample_rate != self.sample_rate:
            mix, clean, estimate = self.resample(
                mix, clean, estimate, fs_from=sample_rate, fs_to=self.sample_rate
            )
        local_mix_counter = Counter()
        local_clean_counter = Counter()
        local_est_counter = Counter()
        # Count the mixture output for each speaker
        txt = self.predict_hypothesis(mix)
        for tmp_id in wav_id:
            out_count = Counter(self.hsdi(truth=self.trans_dic[tmp_id], hypothesis=txt))
            self.mix_counter += out_count
            local_mix_counter += out_count
            self.input_txt_list.append(dict(utt_id=tmp_id, text=txt))
        # Average WER for the clean pair
        for wav, tmp_id in zip(clean, wav_id):
            txt = self.predict_hypothesis(wav)
            out_count = Counter(self.hsdi(truth=self.trans_dic[tmp_id], hypothesis=txt))
            self.clean_counter += out_count
            local_clean_counter += out_count
            self.clean_txt_list.append(dict(utt_id=tmp_id, text=txt))
        # Average WER for the estimate pair
        for est, tmp_id in zip(estimate, wav_id):
            txt = self.predict_hypothesis(est)
            out_count = Counter(self.hsdi(truth=self.trans_dic[tmp_id], hypothesis=txt))
            self.est_counter += out_count
            local_est_counter += out_count
            self.output_txt_list.append(dict(utt_id=tmp_id, text=txt))
        return dict(
            input_wer=self.wer_from_hsdi(**dict(local_mix_counter)),
            clean_wer=self.wer_from_hsdi(**dict(local_clean_counter)),
            wer=self.wer_from_hsdi(**dict(local_est_counter)),
        )

    @staticmethod
    def wer_from_hsdi(hits=0, substitutions=0, deletions=0, insertions=0):
        wer = (substitutions + deletions + insertions) / (hits + substitutions + deletions)
        return wer

    @staticmethod
    def hsdi(truth, hypothesis):
        from jiwer import compute_measures

        keep = ["hits", "substitutions", "deletions", "insertions"]
        out = compute_measures(truth=truth, hypothesis=hypothesis).items()
        return {k: v for k, v in out if k in keep}

    def predict_hypothesis(self, wav):
        nbests = self.asr_model(wav)
        text, *_ = nbests[0]
        return text

    @staticmethod
    def resample(*wavs: np.ndarray, fs_from=None, fs_to=None):
        from resampy import resample as _resample

        return [_resample(w, sr_orig=fs_from, sr_new=fs_to) for w in wavs]

    @staticmethod
    def _df_to_dict(df):
        return {k: v for k, v in zip(df["utt_id"].to_list(), df["text"].to_list())}

    def final_df(self):
        """Generate a MarkDown table, as done by ESPNet."""
        mix_n_word = sum(self.mix_counter[k] for k in ["hits", "substitutions", "deletions"])
        clean_n_word = sum(self.clean_counter[k] for k in ["hits", "substitutions", "deletions"])
        est_n_word = sum(self.est_counter[k] for k in ["hits", "substitutions", "deletions"])
        mix_wer = self.wer_from_hsdi(**dict(self.mix_counter))
        clean_wer = self.wer_from_hsdi(**dict(self.clean_counter))
        est_wer = self.wer_from_hsdi(**dict(self.est_counter))

        mix_hsdi = [
            self.mix_counter[k] for k in ["hits", "substitutions", "deletions", "insertions"]
        ]
        clean_hsdi = [
            self.clean_counter[k] for k in ["hits", "substitutions", "deletions", "insertions"]
        ]
        est_hsdi = [
            self.est_counter[k] for k in ["hits", "substitutions", "deletions", "insertions"]
        ]
        #                   Snt               Wrd         HSDI       Err     S.Err
        for_mix = [len(self.mix_counter), mix_n_word] + mix_hsdi + [mix_wer, "-"]
        for_clean = [len(self.clean_counter), clean_n_word] + clean_hsdi + [clean_wer, "-"]
        for_est = [len(self.est_counter), est_n_word] + est_hsdi + [est_wer, "-"]

        table = [
            ["test_clean / mixture"] + for_mix,
            ["test_clean / clean"] + for_clean,
            ["test_clean / separated"] + for_est,
        ]
        df = pd.DataFrame(
            table, columns=["dataset", "Snt", "Wrd", "Corr", "Sub", "Del", "Ins", "Err", "S.Err"]
        )
        return df

    def final_report_as_markdown(self):
        return self.final_df().to_markdown(index=False, tablefmt="github")
