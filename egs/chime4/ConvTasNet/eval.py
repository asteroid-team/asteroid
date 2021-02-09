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

from asteroid.data.chime4_dataset import CHiME4Dataset
from asteroid import ConvTasNet
from asteroid.models import save_publishable
from asteroid.utils import tensors_to_device
from asteroid.metrics import WERTracker, MockWERTracker

parser = argparse.ArgumentParser()
parser.add_argument(
    "--test_dir", type=str, required=True, help="Test directory including the csv files"
)

parser.add_argument(
    "--use_gpu", type=int, default=0, help="Whether to use the GPU for model execution"
)
parser.add_argument("--exp_dir", default="exp/tmp", help="Experiment root")
parser.add_argument(
    "--n_save_ex", type=int, default=1, help="Number of audio examples to save, -1 means all"
)
parser.add_argument(
    "--compute_wer", type=int, default=1, help="Compute WER using ESPNet's pretrained model"
)
parser.add_argument(
    "--asr_type",
    default="noisy",
    help="Choice for the ASR model whether trained on clean or noisy data. One of clean or noisy",
)


# In CHiME 4 only the noisy data are available, hence no metrics.
COMPUTE_METRICS = []


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


def main(conf):

    if conf["asr_type"] == "noisy":
        asr_model_path = (
            "kamo-naoyuki/chime4_asr_train_asr_transformer3_raw_en_char_sp_valid.acc.ave"
        )
    else:
        asr_model_path = "kamo-naoyuki/wsj_transformer2"

    compute_metrics = update_compute_metrics(conf["compute_wer"], COMPUTE_METRICS)
    annot_path = [f for f in os.listdir(conf["test_dir"]) if "annotations" in f][0]
    anno_df = pd.read_csv(os.path.join(conf["test_dir"], annot_path))
    wer_tracker = (
        MockWERTracker() if not conf["compute_wer"] else WERTracker(asr_model_path, anno_df)
    )
    model_path = os.path.join(conf["exp_dir"], "best_model.pth")
    model = ConvTasNet.from_pretrained(model_path)
    # Handle device placement
    if conf["use_gpu"]:
        model.cuda()
    model_device = next(model.parameters()).device
    test_set = CHiME4Dataset(
        csv_dir=conf["test_dir"],
        sample_rate=conf["sample_rate"],
        segment=None,
        return_id=True,
    )  # Uses all segment length
    # Used to reorder sources only

    # Randomly choose the indexes of sentences to save.
    eval_save_dir = os.path.join(conf["exp_dir"], "chime4", conf["asr_type"])
    ex_save_dir = os.path.join(eval_save_dir, "examples/")
    if conf["n_save_ex"] == -1:
        conf["n_save_ex"] = len(test_set)
    save_idx = random.sample(range(len(test_set)), conf["n_save_ex"])
    series_list = []
    torch.no_grad().__enter__()
    for idx in tqdm(range(len(test_set))):
        # Forward the network on the mixture.
        mix, ids = test_set[idx]
        mix = tensors_to_device(mix, device=model_device)
        est_sources = model(mix.unsqueeze(0))
        mix_np = mix.cpu().data.numpy()
        est_sources_np = est_sources.squeeze(0).cpu().data.numpy()
        est_sources_np *= np.max(np.abs(mix_np)) / np.max(np.abs(est_sources_np))
        # For each utterance, we get a dictionary with the mixture path,
        # the input and output metrics
        utt_metrics = {"mix_path": test_set.mixture_path}
        utt_metrics.update(
            **wer_tracker(
                mix=mix_np,
                clean=None,
                estimate=est_sources_np,
                wav_id=ids,
                sample_rate=conf["sample_rate"],
            )
        )
        series_list.append(pd.Series(utt_metrics))

        # Save some examples in a folder. Wav files and metrics as text.
        if idx in save_idx:
            local_save_dir = os.path.join(ex_save_dir, "ex_{}/".format(idx))
            os.makedirs(local_save_dir, exist_ok=True)
            sf.write(local_save_dir + "mixture.wav", mix_np, conf["sample_rate"])
            # Loop over the sources and estimates
            for src_idx, est_src in enumerate(est_sources_np):
                # est_src *= np.max(np.abs(mix_np)) / np.max(np.abs(est_src))
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
        input_metric_name = "input_" + metric_name
        ldf = all_metrics_df[metric_name] - all_metrics_df[input_metric_name]
        final_results[metric_name] = all_metrics_df[metric_name].mean()
        final_results[metric_name + "_imp"] = ldf.mean()

    print("Overall metrics :")
    pprint(final_results)
    if conf["compute_wer"]:
        print("\nWER report")
        wer_card = wer_tracker.final_report_as_markdown()
        print(wer_card)
        # Save the report
        with open(os.path.join(eval_save_dir, "final_wer.md"), "w") as f:
            f.write(wer_card)
        all_transcriptions = wer_tracker.trans_dic
        with open(os.path.join(eval_save_dir, "all_transcriptions.json"), "w") as f:
            json.dump(all_transcriptions, f, indent=4)

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
    main(arg_dic)
