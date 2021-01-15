import os
import random
import soundfile as sf
import torch
import yaml
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from asteroid.data.chime4_dataset import CHiME4
from asteroid import ConvTasNet
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
    "--n_save_ex", type=int, default=10, help="Number of audio examples to save, -1 means all"
)
parser.add_argument(
    "--compute_wer", type=int, default=0, help="Compute WER using ESPNet's pretrained model"
)

ASR_MODEL_PATH = (
    "kamo-naoyuki/chime4_asr_train_asr_transformer3_raw_en_char_sp_valid.acc.ave"
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


def main(conf):
    compute_metrics = update_compute_metrics(conf["compute_wer"])
    anno_df = pd.read_csv(Path(conf["test_dir"]).parent.parent.parent / "test_annotations.csv")
    wer_tracker = (
        MockWERTracker() if not conf["compute_wer"] else WERTracker(ASR_MODEL_PATH, anno_df)
    )
    model_path = os.path.join(conf["exp_dir"], "best_model.pth")
    model = ConvTasNet.from_pretrained(model_path)
    # Handle device placement
    if conf["use_gpu"]:
        model.cuda()
    model_device = next(model.parameters()).device
    test_set = CHiME4(
        csv_dir=conf["test_dir"],
        sample_rate=conf["sample_rate"],
        segment=None,
    )  # Uses all segment length
    # Used to reorder sources only

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
        mix = test_set[idx]
        mix = mix.to(model_device)
        est_sources = model(mix.unsqueeze(0))
        mix_np = mix.cpu().data.numpy()
        est_sources_np = est_sources.squeeze(0).cpu().data.numpy()

        # Save some examples in a folder. Wav files and metrics as text.
        if idx in save_idx:
            local_save_dir = os.path.join(ex_save_dir, "ex_{}/".format(idx))
            os.makedirs(local_save_dir, exist_ok=True)
            sf.write(local_save_dir + "mixture.wav", mix_np, conf["sample_rate"])
            # Loop over the sources and estimates
            for src_idx, est_src in enumerate(est_sources_np):
                est_src *= np.max(np.abs(mix_np)) / np.max(np.abs(est_src))
                sf.write(
                    local_save_dir + "s{}_estimate.wav".format(src_idx),
                    est_src,
                    conf["sample_rate"],
                )

    if conf["compute_wer"]:
        print("\nWER report")
        wer_card = wer_tracker.final_report_as_markdown()
        print(wer_card)
        # Save the report
        with open(os.path.join(eval_save_dir, "final_wer.md"), "w") as f:
            f.write(wer_card)


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
