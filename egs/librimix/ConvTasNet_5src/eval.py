import os
import random
import soundfile as sf
import torch
import yaml
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from pathlib import Path
from collections import defaultdict

from asteroid.metrics import get_metrics
from asteroid.data import VariableLibriMix
from asteroid.losses import PITLossWrapper, SilenceRobustPairwiseNegSDR
from asteroid import ConvTasNet
from asteroid.models import save_publishable
from asteroid.utils import tensors_to_device
from asteroid.dsp.normalization import normalize_estimates


parser = argparse.ArgumentParser()
parser.add_argument(
    "--test_dir", type=str, required=True, help="Test directory including the csv files"
)
parser.add_argument(
    "--task", type=str, required=True,
    help="One of `sep_clean` or `sep_noisy`",
)
parser.add_argument(
    "--out_dir", type=str, required=True,
    help="Directory in exp_dir where the eval results will be stored",
)
parser.add_argument(
    "--use_gpu", type=int, default=0, help="Whether to use the GPU for model execution"
)
parser.add_argument("--exp_dir", default="exp/tmp", help="Experiment root")
parser.add_argument(
    "--n_save_ex", type=int, default=10, help="Number of audio examples to save, -1 means all"
)

COMPUTE_METRICS = ["si_sdr", "sdr", "sir", "sar", "stoi"]
SILENCE_THRESHOLD = 1e-5  # Energy threshold to classify a target as silent


def count_active_sources(sources_np, threshold=SILENCE_THRESHOLD):
    """Count the number of active (non-silent) sources."""
    n_active = 0
    for i in range(sources_np.shape[0]):
        energy = np.sum(sources_np[i] ** 2)
        if energy > threshold:
            n_active += 1
    return n_active


def main(conf):
    model_path = os.path.join(conf["exp_dir"], "best_model.pth")
    model = ConvTasNet.from_pretrained(model_path)
    if conf["use_gpu"]:
        model.cuda()
    model_device = next(model.parameters()).device

    n_src = conf["train_conf"]["data"]["n_src"]
    silence_threshold = conf["train_conf"].get("loss", {}).get("silence_threshold", SILENCE_THRESHOLD)

    test_set = VariableLibriMix(
        csv_dirs=conf["test_dir"],
        task=conf["task"],
        sample_rate=conf["sample_rate"],
        n_src=n_src,
        segment=None,
        return_id=True,
    )

    # Use silence-robust loss for reordering
    loss_func = PITLossWrapper(
        SilenceRobustPairwiseNegSDR("sisdr", threshold=silence_threshold),
        pit_from="pw_mtx",
    )

    eval_save_dir = os.path.join(conf["exp_dir"], conf["out_dir"])
    ex_save_dir = os.path.join(eval_save_dir, "examples/")
    os.makedirs(eval_save_dir, exist_ok=True)

    if conf["n_save_ex"] == -1:
        conf["n_save_ex"] = len(test_set)
    save_idx = random.sample(range(len(test_set)), min(conf["n_save_ex"], len(test_set)))

    series_list = []
    # Metrics broken down by number of active sources
    per_nsrc_metrics = defaultdict(list)

    torch.no_grad().__enter__()
    for idx in tqdm(range(len(test_set))):
        mix, sources, ids = test_set[idx]
        mix, sources = tensors_to_device([mix, sources], device=model_device)
        est_sources = model(mix.unsqueeze(0))
        loss, reordered_sources = loss_func(est_sources, sources[None], return_est=True)

        mix_np = mix.cpu().data.numpy()
        sources_np = sources.cpu().data.numpy()
        est_sources_np = reordered_sources.squeeze(0).cpu().data.numpy()

        n_active = count_active_sources(sources_np, threshold=silence_threshold)

        # Compute metrics only on active source channels
        active_sources = []
        active_estimates = []
        for i in range(n_src):
            energy = np.sum(sources_np[i] ** 2)
            if energy > silence_threshold:
                active_sources.append(sources_np[i])
                active_estimates.append(est_sources_np[i])

        utt_metrics = {}
        if active_sources:
            active_sources_np = np.stack(active_sources)
            active_estimates_np = np.stack(active_estimates)
            utt_metrics = get_metrics(
                mix_np,
                active_sources_np,
                active_estimates_np,
                sample_rate=conf["sample_rate"],
                metrics_list=COMPUTE_METRICS,
            )

        # Log residual energy in silent output channels
        silent_channel_energies = []
        for i in range(n_src):
            energy = np.sum(sources_np[i] ** 2)
            if energy <= silence_threshold:
                est_energy = np.sum(est_sources_np[i] ** 2)
                silent_channel_energies.append(float(est_energy))

        utt_metrics["n_active_sources"] = n_active
        utt_metrics["mix_path"] = test_set.mixture_path
        if silent_channel_energies:
            utt_metrics["mean_silent_channel_energy"] = np.mean(silent_channel_energies)
            utt_metrics["max_silent_channel_energy"] = np.max(silent_channel_energies)
        else:
            utt_metrics["mean_silent_channel_energy"] = 0.0
            utt_metrics["max_silent_channel_energy"] = 0.0

        series_list.append(pd.Series(utt_metrics))
        per_nsrc_metrics[n_active].append(utt_metrics)

        # Save some examples
        if idx in save_idx:
            local_save_dir = os.path.join(ex_save_dir, f"ex_{idx}/")
            os.makedirs(local_save_dir, exist_ok=True)
            sf.write(os.path.join(local_save_dir, "mixture.wav"), mix_np, conf["sample_rate"])
            est_sources_np_normalized = normalize_estimates(est_sources_np, mix_np)
            for src_idx in range(n_src):
                src_energy = np.sum(sources_np[src_idx] ** 2)
                if src_energy > silence_threshold:
                    sf.write(
                        os.path.join(local_save_dir, f"s{src_idx}.wav"),
                        sources_np[src_idx],
                        conf["sample_rate"],
                    )
                sf.write(
                    os.path.join(local_save_dir, f"s{src_idx}_estimate.wav"),
                    est_sources_np_normalized[src_idx],
                    conf["sample_rate"],
                )
            with open(os.path.join(local_save_dir, "metrics.json"), "w") as f:
                json.dump(utt_metrics, f, indent=0)

    # Save all metrics
    all_metrics_df = pd.DataFrame(series_list)
    all_metrics_df.to_csv(os.path.join(eval_save_dir, "all_metrics.csv"))

    # Print overall metrics
    final_results = {}
    for metric_name in COMPUTE_METRICS:
        if metric_name in all_metrics_df.columns:
            final_results[metric_name] = all_metrics_df[metric_name].mean()
            input_name = f"input_{metric_name}"
            if input_name in all_metrics_df.columns:
                ldf = all_metrics_df[metric_name] - all_metrics_df[input_name]
                final_results[f"{metric_name}_imp"] = ldf.mean()

    final_results["mean_silent_channel_energy"] = all_metrics_df["mean_silent_channel_energy"].mean()
    final_results["max_silent_channel_energy"] = all_metrics_df["max_silent_channel_energy"].max()

    print("\nOverall metrics:")
    pprint(final_results)

    # Print per-source-count breakdown
    print("\nMetrics by number of active sources:")
    for n_active in sorted(per_nsrc_metrics.keys()):
        entries = per_nsrc_metrics[n_active]
        print(f"\n  {n_active}-speaker ({len(entries)} utterances):")
        for metric_name in COMPUTE_METRICS:
            vals = [e[metric_name] for e in entries if metric_name in e]
            if vals:
                print(f"    {metric_name}: {np.mean(vals):.4f}")
        silent_energies = [e["mean_silent_channel_energy"] for e in entries]
        print(f"    mean_silent_channel_energy: {np.mean(silent_energies):.6f}")

    with open(os.path.join(eval_save_dir, "final_metrics.json"), "w") as f:
        json.dump(final_results, f, indent=0)

    model_dict = torch.load(model_path, map_location="cpu")
    os.makedirs(os.path.join(conf["exp_dir"], "publish_dir"), exist_ok=True)
    save_publishable(
        os.path.join(conf["exp_dir"], "publish_dir"),
        model_dict,
        metrics=final_results,
        train_conf=conf["train_conf"],
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
