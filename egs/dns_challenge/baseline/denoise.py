import glob
import os

import soundfile as sf
import torch
import yaml
import json
import argparse
from tqdm import tqdm

from model import load_best_model

parser = argparse.ArgumentParser()
parser.add_argument(
    "--denoise_path", type=str, required=True, help="Directory containing wav files, or file path"
)
parser.add_argument(
    "--use_gpu", type=int, default=0, help="Whether to use the GPU for model execution"
)
parser.add_argument("--exp_dir", default="exp/tmp", help="Experiment root")


def main(conf):
    # Get best trained model
    model = load_best_model(conf["train_conf"], conf["exp_dir"])
    if conf["use_gpu"]:
        model = model.cuda()
    model_device = next(model.parameters()).device
    # Get a list of wav files (or single wav file)
    save_folder = os.path.join(conf["exp_dir"], "denoise")
    os.makedirs(save_folder, exist_ok=True)
    if os.path.isfile(conf["denoise_path"]):
        all_wavs = [conf["denoise_path"]]
    else:
        # If this is a bunch of files we need to denoise, call the subdir
        # of denoise the same way as the basename of the denoise dir.
        save_folder = os.path.join(save_folder, os.path.basename(conf["denoise_path"]))
        all_wavs = glob.glob(conf["denoise_path"] + "*.wav")

    for wav_path in tqdm(all_wavs):
        mix, fs = sf.read(wav_path, dtype="float32")
        with torch.no_grad():
            net_inp = torch.tensor(mix)[None].to(model_device)
            estimate = model.denoise(net_inp).squeeze().cpu().data.numpy()
        # Save the estimate speech
        wav_name = os.path.basename(wav_path)
        sf.write(os.path.join(save_folder, wav_name), estimate, fs)


if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))

    # Load training config
    conf_path = os.path.join(args.exp_dir, "conf.yml")
    with open(conf_path) as conf_file:
        train_conf = yaml.safe_load(conf_file)
    arg_dic["train_conf"] = train_conf

    main(arg_dic)
