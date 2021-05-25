import os
import random
import soundfile as sf
import torch
import yaml
import json
import argparse
import pandas as pd
import numpy as np
from utils import compute_sdr, MAX_INT16, center_trim
from tqdm import tqdm
from pprint import pprint

from conv_tasnet import TasNet
from asteroid.metrics import get_metrics
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from local.tac_dataset import TACDataset
from asteroid.models import save_publishable
from asteroid.utils import tensors_to_device
from deepbeam import OnlineSimulationDataset,vctk_audio, truncator, ms_snsd, simulation_config_test
from preprocess import Prep
from scipy.io.wavfile import write

parser = argparse.ArgumentParser()

parser.add_argument(
    "--use_gpu", type=int, default=1, help="Whether to use the GPU for model execution"
)
parser.add_argument("--exp_dir", default="exp/tmp", help="Experiment root")
parser.add_argument(
    "--n_save_ex", type=int, default=50, help="Number of audio examples to save, -1 means all"
)

compute_metrics = ["si_sdr"]  # , "sdr", "sir", "sar", "stoi"]


def main(conf):
    model_path = os.path.join(conf["exp_dir"], "best_model.ckpt")
    #model_path = "/tmp/pycharm_project_591/exp/tmp/checkpoints/epoch=8-step=2519.ckpt"
    pretrain = torch.load(model_path, map_location="cpu")
    model = TasNet()
    model.load_state_dict(pretrain)
    conf["use_gpu"] = False
    # Handle device placement
    if conf["use_gpu"]:
        model.cuda()
    model_device = next(model.parameters()).device
    #test_set = TACDataset(args.test_json, train=False)
    test_set = OnlineSimulationDataset(vctk_audio, ms_snsd, 48, simulation_config_test, truncator, "./test_online", 50)

    # Used to reorder sources only
    #loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")

    # Randomly choose the indexes of sentences to save.
    torch.no_grad().__enter__()
    input_sdr_list = []
    output_sdr_list = []
    model.eval()
    for idx in tqdm(range(len(test_set))):

        # Forward the network on the mixture.
        input = test_set.__getitem__(idx)
        mix = input[0]
        fusion = Prep(input)
        mix = np.expand_dims(mix, axis=0)  # 1 * channel * length
        mix = torch.from_numpy(mix).to(model_device).float()
        ref = input[3] * MAX_INT16
        #raw = torch.tensor(mix, dtype=torch.float32, device=model_device)
        ref = torch.tensor(ref, dtype=torch.float32, device=model_device)



        #valid_mics = torch.ones((len(mix), 1)).to(dtype=torch.long, device=raw.device)
        est_list = []
        for i in range(conf['train_conf']['net']['n_src']):
            est = model(mix, fusion[i])
            est_list.append(est)

        spks = torch.cat(est_list, dim=1)
        ref = center_trim(ref, spks).transpose(1, 0)
        #loss, spks = loss_func(spks, ref, return_est=True)
        spks = spks.data.cpu().numpy().squeeze()
        ref = ref.data.cpu().numpy()

        for idx, samps in enumerate(spks):
            samps = samps * MAX_INT16
            input_sdr_list.append(compute_sdr(ref[0, idx], mix[0, 0, :] * MAX_INT16))
            output_sdr_list.append(compute_sdr(ref[0, idx], samps))
    input_sdr_array = np.array(input_sdr_list)
    output_sdr_array = np.array(output_sdr_list)
    result = np.median(output_sdr_array - input_sdr_array)
    print("The SNR: " + str(result))


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