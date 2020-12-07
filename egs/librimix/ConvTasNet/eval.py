import os
import random
from typing import List
import soundfile as sf
import torch
import yaml
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from collections import Counter
from pathlib import Path

from asteroid.metrics import get_metrics
from asteroid.data.librimix_dataset import LibriMix
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid import ConvTasNet
from asteroid.models import save_publishable
from asteroid.utils import tensors_to_device


def import_compute_measures():
    try:
        from jiwer import compute_measures

        # FIXME
        # Which returns
        #  {
        #     "wer": wer,
        #     "mer": mer,
        #     "wil": wil,
        #     "wip": wip,
        #     "hits": H,
        #     "substitutions": S,
        #     "deletions": D,
        #     "insertions": I,
        # }
    except ModuleNotFoundError:
        return None
    else:
        return compute_measures


compute_measures = import_compute_measures()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--test_dir", type=str, required=True, help="Test directory including the csv files"
)
parser.add_argument(
    "--task",
    type=str,
    required=True,
    help="One of `enh_single`, `enh_both`, " "`sep_clean` or `sep_noisy`",
)
parser.add_argument(
    "--out_dir",
    type=str,
    required=True,
    help="Directory in exp_dir where the eval results" " will be stored",
)
parser.add_argument(
    "--use_gpu", type=int, default=0, help="Whether to use the GPU for model execution"
)
parser.add_argument("--exp_dir", default="exp/tmp", help="Experiment root")
parser.add_argument(
    "--n_save_ex", type=int, default=10, help="Number of audio examples to save, -1 means all"
)
parser.add_argument(
    "--compute_wer", type=int, default=0, help="Compute WER using ESPNet's pretrained model"
)

COMPUTE_METRICS = ["si_sdr", "sdr", "sir", "sar", "stoi"]
ASR_MODEL_PATH = (
    "Shinji Watanabe/librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best"
)


def update_compute_metrics(compute_wer, metric_list):
    if not compute_wer:
        return metric_list
    try:
        from espnet2.bin.asr_inference import Speech2Text
        from espnet_model_zoo.downloader import ModelDownloader
    except ModuleNotFoundError:
        import warnings
        warnings.warn("Couldn't find espnet installation. Continuing without.")
        return metric_list
    return metric_list + ["wer"]


class MockTracker:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return dict()


class WERTracker:
    def __init__(self, model_name, trans_df):
        from espnet2.bin.asr_inference import Speech2Text
        from espnet_model_zoo.downloader import ModelDownloader

        self.model_name = model_name
        d = ModelDownloader()
        self.asr_model = Speech2Text(**d.download_and_unpack(model_name))
        self.input_txt_list = []
        self.output_txt_list = []
        self.sample_rate = int(d.data_frame[d.data_frame["name"] == model_name]["fs"])
        self.trans_df = trans_df
        self.trans_dic = self._df_to_dict(trans_df)
        # FIXME, set jiwer transform here for text preprocessing.
        self.mix_counter = Counter()
        self.est_counter = Counter()

    def __call__(
        self, *, mix: np.ndarray, est_sources: np.ndarray, sample_rate: int, wav_id: List[str]
    ):
        """Compute and store best hypothesis for the mixture and the estimates"""
        if sample_rate != self.sample_rate:
            mix, est_sources = self.resample(
                mix, est_sources, fs_from=sample_rate, fs_to=self.sample_rate
            )
        # FIXME: compute metrics on the references signals as well?
        local_mix_counter = Counter()
        local_est_counter = Counter()
        # Count the mixture output for each speaker
        txt = self.predict_hypothesis(mix)
        for tmp_id in wav_id:
            out_count = Counter(self.hsdi(truth=self.trans_dic[tmp_id], hypothesis=txt))
            self.mix_counter += out_count
            local_mix_counter += out_count
            self.input_txt_list.append(dict(utt_id=tmp_id, text=txt))
        # Average WER for the estimate pair
        for est, tmp_id in zip(est_sources, wav_id):
            txt = self.predict_hypothesis(est)
            out_count = Counter(self.hsdi(truth=self.trans_dic[tmp_id], hypothesis=txt))
            self.est_counter += out_count
            local_est_counter += out_count
            self.output_txt_list.append(dict(utt_id=tmp_id, text=txt))
        return dict(
            input_wer=self.wer_from_hsdi(**dict(local_mix_counter)),
            wer=self.wer_from_hsdi(**dict(local_est_counter)),
        )

    @staticmethod
    def wer_from_hsdi(hits=0, substitutions=0, deletions=0, insertions=0):
        wer = (substitutions + deletions + insertions) / (hits + substitutions + deletions)
        return wer

    @staticmethod
    def hsdi(truth, hypothesis):
        keep = ["hits", "substitutions", "deletions", "insertions"]
        out = compute_measures(truth=truth, hypothesis=hypothesis).items()
        return {k: v for k, v in out if k in keep}

    @staticmethod
    def dict_add(d, **kwargs):
        assert d.keys() == kwargs.keys()
        return {k: d[k] + kwargs[k] for k in d.keys()}

    def predict_hypothesis(self, wav):
        nbests = self.asr_model(wav)
        text, *_ = nbests[0]
        return text

    def to_df(self):
        # Should we have that?
        return

    @staticmethod
    def resample(*wavs: np.ndarray, fs_from=None, fs_to=None):
        from resampy import resample as _resample

        return [_resample(w, sr_orig=fs_from, sr_new=fs_to) for w in wavs]

    def _df_to_dict(self, df):
        return {k: v for k, v in zip(df["utt_id"].to_list(), df["text"].to_list())}

    def final_report(self):
        """Generate a MarkDown table, as done by ESPNet."""
        mix_n_word = sum(self.mix_counter[k] for k in ["hits", "substitutions", "deletions"])
        est_n_word = sum(self.est_counter[k] for k in ["hits", "substitutions", "deletions"])
        mix_wer = self.wer_from_hsdi(**dict(self.mix_counter))
        est_wer = self.wer_from_hsdi(**dict(self.est_counter))

        mix_hsdi = [
            self.mix_counter[k] for k in ["hits", "substitutions", "deletions", "insertions"]
        ]
        est_hsdi = [
            self.est_counter[k] for k in ["hits", "substitutions", "deletions", "insertions"]
        ]
        #                   Snt               Wrd         HSDI       Err     S.Err
        for_mix = [len(self.mix_counter), mix_n_word] + mix_hsdi + [mix_wer, "-"]
        for_est = [len(self.est_counter), est_n_word] + est_hsdi + [est_wer, "-"]

        line_list = [
            "| dataset | Snt | Wrd | Corr | Sub | Del | Ins | Err | S.Err |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
            f"| decode_asr_lm / test_clean / mixture   |" + " | ".join(map(str, for_mix)) + "|",
            f"| decode_asr_lm / test_clean / separated |" + " | ".join(map(str, for_est)) + "|",
        ]
        result_card = "\n".join(line_list)
        return result_card


def main(conf):
    compute_metrics = update_compute_metrics(conf["compute_wer"], COMPUTE_METRICS)
    anno_df = pd.read_csv(Path(conf["test_dir"]).parent.parent.parent / "test_annotations.csv")
    wer_tracker = MockTracker() if not conf["compute_wer"] else WERTracker(ASR_MODEL_PATH, anno_df)
    model_path = os.path.join(conf["exp_dir"], "best_model.pth")
    model = ConvTasNet.from_pretrained(model_path)
    # Handle device placement
    if conf["use_gpu"]:
        model.cuda()
    model_device = next(model.parameters()).device
    test_set = LibriMix(
        csv_dir=conf["test_dir"],
        task=conf["task"],
        sample_rate=conf["sample_rate"],
        n_src=conf["train_conf"]["data"]["n_src"],
        segment=None,
        return_id=True,
        # FIXME: ensure max mode for eval.
    )  # Uses all segment length
    # Used to reorder sources only
    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")

    # Randomly choose the indexes of sentences to save.
    eval_save_dir = os.path.join(conf["exp_dir"], conf["out_dir"])
    ex_save_dir = os.path.join(eval_save_dir, "examples/")
    if conf["n_save_ex"] == -1:
        conf["n_save_ex"] = len(test_set)
    save_idx = random.sample(range(len(test_set)), conf["n_save_ex"])
    series_list = []
    torch.no_grad().__enter__()
    for idx in tqdm(range(len(test_set))):
        # Forward the network on the mixture.
        mix, sources, ids = test_set[idx]
        mix, sources = tensors_to_device([mix, sources], device=model_device)
        est_sources = model(mix.unsqueeze(0))
        loss, reordered_sources = loss_func(est_sources, sources[None], return_est=True)
        mix_np = mix.cpu().data.numpy()
        sources_np = sources.cpu().data.numpy()
        est_sources_np = reordered_sources.squeeze(0).cpu().data.numpy()
        # For each utterance, we get a dictionary with the mixture path,
        # the input and output metrics
        utt_metrics = get_metrics(
            mix_np,
            sources_np,
            est_sources_np,
            sample_rate=conf["sample_rate"],
            metrics_list=COMPUTE_METRICS,
        )
        utt_metrics["mix_path"] = test_set.mixture_path
        utt_metrics.update(
            **wer_tracker(
                mix=mix_np, est_sources=sources_np, wav_id=ids, sample_rate=conf["sample_rate"]
            )
        )
        series_list.append(pd.Series(utt_metrics))

        # Save some examples in a folder. Wav files and metrics as text.
        if idx in save_idx:
            local_save_dir = os.path.join(ex_save_dir, "ex_{}/".format(idx))
            os.makedirs(local_save_dir, exist_ok=True)
            sf.write(local_save_dir + "mixture.wav", mix_np, conf["sample_rate"])
            # Loop over the sources and estimates
            for src_idx, src in enumerate(sources_np):
                sf.write(local_save_dir + "s{}.wav".format(src_idx), src, conf["sample_rate"])
            for src_idx, est_src in enumerate(est_sources_np):
                est_src *= np.max(np.abs(mix_np)) / np.max(np.abs(est_src))
                sf.write(
                    local_save_dir + "s{}_estimate.wav".format(src_idx),
                    est_src,
                    conf["sample_rate"],
                )
            # Write local metrics to the example folder.
            with open(local_save_dir + "metrics.json", "w") as f:
                json.dump(utt_metrics, f, indent=0)

    # Save all metrics to the experiment folder.
    all_metrics_df = pd.DataFrame(series_list)
    all_metrics_df.to_csv(os.path.join(eval_save_dir, "all_metrics.csv"))

    # Print and save summary metrics
    final_results = {}
    if conf["compute_wer"]:
        compute_metrics.append("wer")  # Just for the final .json.
    for metric_name in compute_metrics:
        input_metric_name = "input_" + metric_name
        ldf = all_metrics_df[metric_name] - all_metrics_df[input_metric_name]
        final_results[metric_name] = all_metrics_df[metric_name].mean()
        final_results[metric_name + "_imp"] = ldf.mean()

    print("Overall metrics :")
    pprint(final_results)
    if conf["compute_wer"]:
        print("\nWER report")
        wer_card = wer_tracker.final_report()
        pprint(wer_card)

    with open(os.path.join(eval_save_dir, "final_metrics.json"), "w") as f:
        json.dump(final_results, f, indent=0)

    model_dict = torch.load(model_path, map_location="cpu")
    os.makedirs(os.path.join(conf["exp_dir"], "publish_dir"), exist_ok=True)
    publishable = save_publishable(
        os.path.join(conf["exp_dir"], "publish_dir"),
        model_dict,
        metrics=final_results,
        train_conf=train_conf,
    )


if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))
    # Load training config
    conf_path = os.path.join(args.exp_dir, "conf.yml")
    with open(conf_path) as f:
        train_conf = yaml.safe_load(f)
    arg_dic["sample_rate"] = train_conf["data"]["sample_rate"]
    arg_dic["train_conf"] = train_conf

    if args.task != arg_dic["train_conf"]["data"]["task"]:
        print(
            "Warning : the task used to test is different than "
            "the one from training, be sure this is what you want."
        )

    main(arg_dic)
