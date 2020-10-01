import glob
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

from asteroid.metrics import get_metrics

from model import load_best_model
from local.preprocess_dns import make_wav_id_dict

parser = argparse.ArgumentParser()
parser.add_argument(
    "--test_dir", type=str, required=True, help="Test directory including wav files"
)
parser.add_argument(
    "--use_gpu", type=int, default=0, help="Whether to use the GPU for model execution"
)
parser.add_argument("--exp_dir", default="exp/tmp", help="Experiment root")
parser.add_argument(
    "--n_save_ex", type=int, default=50, help="Number of audio examples to save, -1 means all"
)

ALL_METRICS = ["si_sdr", "sdr", "sir", "sar", "stoi", "pesq"]
COMPUTE_METRICS = ALL_METRICS


def main(conf):
    # Get best trained model
    model = load_best_model(conf["train_conf"], conf["exp_dir"])
    if conf["use_gpu"]:
        model = model.cuda()
    # Evaluate performances separately w/ and w/o reverb
    for subdir in ["with_reverb", "no_reverb"]:
        dict_list = get_wavs_dict_list(os.path.join(conf["test_dir"], subdir))
        save_dir = os.path.join(conf["exp_dir"], subdir + "examples/")
        os.makedirs(save_dir, exist_ok=True)
        all_metrics_df = evaluate(dict_list, model, conf=conf, save_dir=save_dir)
        all_metrics_df.to_csv(os.path.join(conf["exp_dir"], "all_metrics_{}.csv".format(subdir)))
        # Print and save summary metrics
        final_results = {}
        for metric_name in COMPUTE_METRICS:
            input_metric_name = "input_" + metric_name
            ldf = all_metrics_df[metric_name] - all_metrics_df[input_metric_name]
            final_results[metric_name] = all_metrics_df[metric_name].mean()
            final_results[metric_name + "_imp"] = ldf.mean()
        print("Overall metrics {} :".format(subdir))
        pprint(final_results)
        filename = os.path.join(conf["exp_dir"], "final_metrics_{}.json".format(subdir))
        with open(filename, "w") as f:
            json.dump(final_results, f, indent=0)


def get_wavs_dict_list(test_dir):
    """Creates a list of example pair dictionaries.

    Args:
        test_dir (str): Directory where clean/ and noisy/ subdirectories can
            be found.
    Returns:
        List[dict] : list of noisy/clean pair dictionaries.
            Each dict looks like :
                {'clean': clean_path,
                'noisy': noisy_path,
                'id': 3}
    """
    # Find all clean files and make an {id: filepath} dictionary
    clean_wavs = glob.glob(os.path.join(test_dir, "clean/*.wav"))
    clean_dic = make_wav_id_dict(clean_wavs)
    # Same for noisy files
    noisy_wavs = glob.glob(os.path.join(test_dir, "noisy/*.wav"))
    noisy_dic = make_wav_id_dict(noisy_wavs)
    assert clean_dic.keys() == noisy_dic.keys()
    # Combine both dictionaries
    dict_list = [dict(clean=clean_dic[k], noisy=noisy_dic[k], id=k) for k in clean_dic.keys()]
    return dict_list


def evaluate(dict_list, model, conf, save_dir=None):
    model_device = next(model.parameters()).device
    # Randomly choose the indexes of sentences to save.
    if save_dir is None:
        conf["n_save_ex"] = 0
    if conf["n_save_ex"] == -1:
        conf["n_save_ex"] = len(dict_list)
    save_idx = random.sample(range(len(dict_list)), conf["n_save_ex"])
    series_list = []
    for idx, wav_dic in enumerate(tqdm(dict_list)):
        # Forward the network on the mixture.
        noisy_np, clean_np, fs = load_wav_dic(wav_dic)
        with torch.no_grad():
            net_input = torch.tensor(noisy_np)[None, None].to(model_device)
            est_clean_np = model.denoise(net_input).squeeze().cpu().data.numpy()

        utt_metrics = get_metrics(
            mix=noisy_np,
            clean=clean_np,
            estimate=est_clean_np,
            sample_rate=fs,
            metrics_list=COMPUTE_METRICS,
        )
        utt_metrics["noisy_path"] = wav_dic["noisy"]
        utt_metrics["clean_path"] = wav_dic["clean"]
        series_list.append(pd.Series(utt_metrics))
        # Save some examples in a folder. Wav files and metrics as text.
        if idx in save_idx:
            local_save_dir = os.path.join(save_dir, "ex_{}/".format(idx))
            os.makedirs(local_save_dir, exist_ok=True)
            sf.write(local_save_dir + "noisy.wav", noisy_np, fs)
            sf.write(local_save_dir + "clean.wav", clean_np, fs)
            sf.write(local_save_dir + "estimate.wav", est_clean_np, fs)
            # Write local metrics to the example folder.
            with open(local_save_dir + "metrics.json", "w") as f:
                json.dump(utt_metrics, f, indent=0)

    # Save all metrics to the experiment folder.
    all_metrics_df = pd.DataFrame(series_list)
    return all_metrics_df


def load_wav_dic(wav_dic):
    """Load wavs files from a dictionary with path entries.

    Returns:
        tuple: noisy speech waveform, clean speech waveform.
    """
    noisy_path, clean_path = wav_dic["noisy"], wav_dic["clean"]
    noisy, fs = sf.read(noisy_path, dtype="float32")
    clean, fs = sf.read(clean_path, dtype="float32")
    return noisy, clean, fs


if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))

    # Load training config
    conf_path = os.path.join(args.exp_dir, "conf.yml")
    with open(conf_path) as conf_file:
        train_conf = yaml.safe_load(conf_file)
    arg_dic["train_conf"] = train_conf

    main(arg_dic)
