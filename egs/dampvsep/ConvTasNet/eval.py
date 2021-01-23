import os
import random
import soundfile as sf
import torch
import yaml
import json
import argparse
import pandas as pd
from tqdm import tqdm
from pprint import pprint
import numpy as np
from pathlib import Path

from asteroid.metrics import get_metrics
from asteroid import ConvTasNet
from asteroid.models import save_publishable
from asteroid.data import DAMPVSEPDataset
from asteroid.dsp import LambdaOverlapAdd


parser = argparse.ArgumentParser()
parser.add_argument(
    "--out_dir",
    type=str,
    required=True,
    help="Directory in exp_dir where the eval results will be stored",
)
parser.add_argument(
    "--use_gpu", type=int, default=0, help="Whether to use the GPU for model execution"
)
parser.add_argument("--exp_dir", default="exp/tmp", help="Experiment root")
parser.add_argument(
    "--n_save_ex", type=int, default=10, help="Number of audio examples to save, -1 means all"
)

compute_metrics = ["si_sdr", "sdr", "sir", "sar", "stoi"]


def main(conf):
    model_path = os.path.join(conf["exp_dir"], "best_model.pth")
    model = ConvTasNet.from_pretrained(model_path)
    model = LambdaOverlapAdd(
        nnet=model,  # function to apply to each segment.
        n_src=2,  # number of sources in the output of nnet
        window_size=64000,  # Size of segmenting window
        hop_size=None,  # segmentation hop size
        window="hanning",  # Type of the window (see scipy.signal.get_window
        reorder_chunks=False,  # Whether to reorder each consecutive segment.
        enable_grad=False,  # Set gradient calculation on of off (see torch.set_grad_enabled)
    )

    # Handle device placement
    if conf["use_gpu"]:
        model.cuda()

    model_device = next(model.parameters()).device

    # Evaluation is mode using 'remix' mixture
    dataset_kwargs = {
        "root_path": Path(conf["train_conf"]["data"]["root_path"]),
        "task": conf["train_conf"]["data"]["task"],
        "sample_rate": conf["train_conf"]["data"]["sample_rate"],
        "num_workers": conf["train_conf"]["training"]["num_workers"],
        "mixture": "remix",
    }

    test_set = DAMPVSEPDataset(split="test", **dataset_kwargs)

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
        mix, sources = test_set[idx]
        mix = mix.to(model_device)
        est_sources = model.forward(mix.unsqueeze(0).unsqueeze(1))
        mix_np = mix.squeeze(0).cpu().data.numpy()
        sources_np = sources.cpu().data.numpy()
        est_sources_np = est_sources.squeeze(0).cpu().data.numpy()

        # For each utterance, we get a dictionary with the mixture path,
        # the input and output metrics
        utt_metrics = get_metrics(
            mix_np,
            sources_np,
            est_sources_np,
            sample_rate=conf["sample_rate"],
            metrics_list=compute_metrics,
            average=False,
        )
        utt_metrics = split_metric_dict(utt_metrics)
        utt_metrics["mix_path"] = test_set.mixture_path
        series_list.append(pd.Series(utt_metrics))
        # Save some examples in a folder. Wav files and metrics as text.
        if idx in save_idx:
            local_save_dir = os.path.join(ex_save_dir, "ex_{}/".format(idx))
            os.makedirs(local_save_dir, exist_ok=True)
            sf.write(local_save_dir + "mixture.wav", mix_np / max(abs(mix_np)), conf["sample_rate"])

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
    for metric_name in compute_metrics:
        for s in ["", "_s0", "_s1"]:
            input_metric_name = "input_" + f"{metric_name}{s}"
            ldf = all_metrics_df[f"{metric_name}{s}"] - all_metrics_df[input_metric_name]
            final_results[f"{metric_name}{s}"] = all_metrics_df[f"{metric_name}{s}"].mean()
            final_results[f"{metric_name}{s}" + "_imp"] = ldf.mean()
    print("Overall metrics :")
    pprint(final_results)
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


def split_metric_dict(dic):
    """Take average of numpy arrays in a dictionary.
    Args:
        dic (dict): Input dictionary to take average from
    Returns:
        dict: New dictionary with array averaged.
    """
    # Copy dic first
    dic2 = dict(dic)
    for k, v in dic.items():
        if isinstance(v, np.ndarray):
            dic2[f"{k}_s0"] = float(v[0])
            dic2[f"{k}_s1"] = float(v[1])
            dic2[k] = float(v.mean())
    return dic2


if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))
    # Load training config
    conf_path = os.path.join(args.exp_dir, "conf.yml")
    with open(conf_path) as f:
        train_conf = yaml.safe_load(f)
    arg_dic["sample_rate"] = train_conf["data"]["sample_rate"]
    arg_dic["train_conf"] = train_conf

    main(arg_dic)
