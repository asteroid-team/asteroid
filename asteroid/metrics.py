import json
import torch
import warnings
import traceback
from typing import List
from collections import Counter
import pandas as pd
import numpy as np
from pb_bss_eval import InputMetrics, OutputMetrics
import torch.nn as nn

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
        metrics_list (Union[List[str], str): List of metrics to compute.
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


class MetricTracker:
    """Metric tracker, subject to change.

    Args:
        sample_rate (int): sampling rate of the audio clips.
        metrics_list (Union[List[str], str): List of metrics to compute.
            Defaults to 'all' (['si_sdr', 'sdr', 'sir', 'sar', 'stoi', 'pesq']).
        average (bool): Return dict([float]) if True, else dict([array]).
        compute_permutation (bool): Whether to compute the permutation on
            estimate sources for the output metrics (default False)
        ignore_metrics_errors (bool): Whether to ignore errors that occur in
            computing the metrics. A warning will be printed instead.
    """

    def __init__(
        self,
        sample_rate,
        metrics_list=tuple(ALL_METRICS),
        average=True,
        compute_permutation=False,
        ignore_metrics_errors=False,
    ):
        self.sample_rate = sample_rate
        # TODO: support WER in metrics_list when merged.
        self.metrics_list = metrics_list
        self.average = average
        self.compute_permutation = compute_permutation
        self.ignore_metrics_errors = ignore_metrics_errors

        self.series_list = []
        self._len_last_saved = 0
        self._all_metrics = pd.DataFrame()

    def __call__(
        self, *, mix: np.ndarray, clean: np.ndarray, estimate: np.ndarray, filename=None, **kwargs
    ):
        """Compute metrics for mix/clean/estimate and log it to the class.

        Args:
            mix (np.array): mixture array.
            clean (np.array): reference array.
            estimate (np.array): estimate array.
            sample_rate (int): sampling rate of the audio clips.
            filename (str, optional): If computing a metric fails, print this
                filename along with the exception/warning message for debugging purposes.
            **kwargs: Any key, value pair to log in the utterance metric (filename, speaker ID, etc...)
        """
        utt_metrics = get_metrics(
            mix,
            clean,
            estimate,
            sample_rate=self.sample_rate,
            metrics_list=self.metrics_list,
            average=self.average,
            compute_permutation=self.compute_permutation,
            ignore_metrics_errors=self.ignore_metrics_errors,
            filename=filename,
        )
        utt_metrics.update(kwargs)
        self.series_list.append(pd.Series(utt_metrics))

    def as_df(self):
        """Return dataframe containing the results (cached)."""
        if self._len_last_saved == len(self.series_list):
            return self._all_metrics
        self._len_last_saved = len(self.series_list)
        self._all_metrics = pd.DataFrame(self.series_list)
        return pd.DataFrame(self.series_list)

    def final_report(self, dump_path: str = None):
        """Return dict of average metrics. Dump to JSON if `dump_path` is not None."""
        final_results = {}
        metrics_df = self.as_df()
        for metric_name in self.metrics_list:
            input_metric_name = "input_" + metric_name
            ldf = metrics_df[metric_name] - metrics_df[input_metric_name]
            final_results[metric_name] = metrics_df[metric_name].mean()
            final_results[metric_name + "_imp"] = ldf.mean()
        if dump_path is not None:
            dump_path = dump_path + ".json" if not dump_path.endswith(".json") else dump_path
            with open(dump_path, "w") as f:
                json.dump(final_results, f, indent=0)
        return final_results


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
        use_gpu (bool): Whether to use GPU for forward caculation.
    """

    def __init__(self, model_name, trans_df, use_gpu=True):

        from espnet2.bin.asr_inference import Speech2Text
        from espnet_model_zoo.downloader import ModelDownloader
        import jiwer

        self.model_name = model_name
        self.device = "cuda" if use_gpu else "cpu"
        d = ModelDownloader()
        self.asr_model = Speech2Text(**d.download_and_unpack(model_name), device=self.device)
        self.input_txt_list = []
        self.clean_txt_list = []
        self.output_txt_list = []
        self.transcriptions = []
        self.true_txt_list = []
        self.sample_rate = int(d.data_frame[d.data_frame["name"] == model_name]["fs"])
        self.trans_df = trans_df
        self.trans_dic = self._df_to_dict(trans_df)
        self.mix_counter = Counter()
        self.clean_counter = Counter()
        self.est_counter = Counter()
        self.transformation = jiwer.Compose(
            [
                jiwer.ToLowerCase(),
                jiwer.RemovePunctuation(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
                jiwer.SentencesToListOfWords(),
                jiwer.RemoveEmptyStrings(),
            ]
        )

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

        # Dict to gather transcriptions and IDs
        trans_dict = dict(mixture_txt={}, clean={}, estimates={}, truth={})
        # Get mixture transcription
        trans_dict["mixture_txt"] = txt
        #  Get ground truth transcription and IDs
        for i, tmp_id in enumerate(wav_id):
            trans_dict["truth"][f"utt_id_{i}"] = tmp_id
            trans_dict["truth"][f"txt_{i}"] = self.trans_dic[tmp_id]
            self.true_txt_list.append(dict(utt_id=tmp_id, text=self.trans_dic[tmp_id]))
        # Mixture
        for tmp_id in wav_id:
            out_count = Counter(
                self.hsdi(
                    truth=self.trans_dic[tmp_id], hypothesis=txt, transformation=self.transformation
                )
            )
            self.mix_counter += out_count
            local_mix_counter += out_count
            self.input_txt_list.append(dict(utt_id=tmp_id, text=txt))
        # Average WER for the clean pair
        for i, (wav, tmp_id) in enumerate(zip(clean, wav_id)):
            txt = self.predict_hypothesis(wav)
            out_count = Counter(
                self.hsdi(
                    truth=self.trans_dic[tmp_id], hypothesis=txt, transformation=self.transformation
                )
            )
            self.clean_counter += out_count
            local_clean_counter += out_count
            self.clean_txt_list.append(dict(utt_id=tmp_id, text=txt))
            trans_dict["clean"][f"utt_id_{i}"] = tmp_id
            trans_dict["clean"][f"txt_{i}"] = txt
        # Average WER for the estimate pair
        for i, (est, tmp_id) in enumerate(zip(estimate, wav_id)):
            txt = self.predict_hypothesis(est)
            out_count = Counter(
                self.hsdi(
                    truth=self.trans_dic[tmp_id], hypothesis=txt, transformation=self.transformation
                )
            )
            self.est_counter += out_count
            local_est_counter += out_count
            self.output_txt_list.append(dict(utt_id=tmp_id, text=txt))
            trans_dict["estimates"][f"utt_id_{i}"] = tmp_id
            trans_dict["estimates"][f"txt_{i}"] = txt
        self.transcriptions.append(trans_dict)
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
    def hsdi(truth, hypothesis, transformation):
        from jiwer import compute_measures

        keep = ["hits", "substitutions", "deletions", "insertions"]
        out = compute_measures(
            truth=truth,
            hypothesis=hypothesis,
            truth_transform=transformation,
            hypothesis_transform=transformation,
        ).items()
        return {k: v for k, v in out if k in keep}

    def predict_hypothesis(self, wav):
        wav = torch.from_numpy(wav).to(self.device)
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


class F1Tracker(nn.Module):
    """F1 score tracker."""

    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        tp = torch.sum(torch.logical_and(y_pred, y_true))
        tn = torch.sum(torch.logical_and(torch.logical_not(y_pred), torch.logical_not(y_true)))
        fp = torch.sum(torch.logical_and(torch.logical_xor(y_pred, y_true), y_pred))
        fn = torch.sum(torch.logical_and(torch.logical_xor(y_pred, y_true), y_true))
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        }
