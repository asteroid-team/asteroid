import os
import random

import matplotlib.pyplot as plt
import torch
import json
import argparse
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from asteroid.metrics import F1Tracker
from asteroid.binarize import Binarize

from asteroid.models.conv_tasnet import VADNet
from asteroid.data.vad_dataset import LibriVADDataset

parser = argparse.ArgumentParser()
parser.add_argument(
    "--md_path", type=str, required=True, help="Test directory including the csv files"
)
parser.add_argument(
    "--out_dir",
    type=str,
    required=True,
    help="Directory in exp_dir where the eval results will be stored",
)
parser.add_argument(
    "--exp_dir",
    type=str,
    required=True,
    help="Directory of the exp",
)

parser.add_argument(
    "--n_save_ex", type=int, default=10, help="Number of audio examples to save, -1 means all"
)

parser.add_argument("--threshold", type=float, default=0.5)
compute_metrics = ["accuracy", "precision", "recall", "f1_score"]


def main(conf):
    test_set = LibriVADDataset(md_file_path=conf["md_path"], segment=None)
    model = VADNet.from_pretrained(os.path.join(conf["exp_dir"], "best_model.pth"))
    # Used to reorder sources only
    # Randomly choose the indexes of sentences to save.
    eval_save_dir = os.path.join(conf["exp_dir"], conf["out_dir"])
    ex_save_dir = os.path.join(eval_save_dir, "examples/")
    if conf["n_save_ex"] == -1:
        conf["n_save_ex"] = len(test_set)
    save_idx = random.sample(range(len(test_set)), conf["n_save_ex"])
    series_list = []
    torch.no_grad().__enter__()
    tracker = F1Tracker()
    binarizer = Binarize(threshold=conf["threshold"], stability=0.05)

    for idx in tqdm(range(len(test_set))):
        # Forward the network on the mixture.
        mix, labels = test_set[idx]
        est = model(mix.unsqueeze(0))
        binarized = binarizer(est)
        utt_metrics = tracker(binarized, labels)
        utt_metrics["source_path"] = test_set.source_path
        series_list.append(pd.Series(utt_metrics))

        # Save some examples in a folder. Wav files and metrics as text.
        if idx in save_idx:
            local_save_dir = os.path.join(ex_save_dir, "ex_{}/".format(idx))
            os.makedirs(local_save_dir, exist_ok=True)
            # Write local metrics to the example folder.
            # Create two subplots and unpack the output array immediately
            fig, axs = plt.subplots(3, sharex=True, sharey=True)
            axs[0].plot(labels.squeeze().data.numpy())
            axs[1].plot(binarized.squeeze().data.numpy())
            axs[2].plot(est.squeeze().data.numpy())
            axs[0].title.set_text("Ground truth")
            axs[1].title.set_text("Estimate")
            axs[2].title.set_text("Raw")
            plt.savefig(os.path.join(local_save_dir, "result.png"))
            with open(local_save_dir + "metrics.json", "w") as f:
                json.dump(utt_metrics, f, indent=0)

    # Save all metrics to the experiment folder.
    all_metrics_df = pd.DataFrame(series_list)
    all_metrics_df.to_csv(os.path.join(eval_save_dir, "all_metrics.csv"))
    all_metrics_df = pd.read_csv(os.path.join(eval_save_dir, "all_metrics.csv"))

    # Print and save summary metrics
    final_results = {}
    for metric_name in compute_metrics:
        final_results[metric_name] = all_metrics_df[metric_name].mean()

    print("Overall metrics :")
    pprint(final_results)

    with open(os.path.join(eval_save_dir, "final_metrics.json"), "w") as f:
        json.dump(final_results, f, indent=0)  # Save all metrics to the experiment folder.


if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))
    main(arg_dic)
