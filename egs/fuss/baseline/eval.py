import os
import random
import soundfile as sf
import torch
import yaml
import json
import argparse
import pandas as pd
from asteroid.metrics import get_metrics
from tqdm import tqdm
from pprint import pprint
from pathlib import Path

from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr, singlesrc_neg_sisdr
from asteroid.data.wham_dataset import WhamDataset
from asteroid.utils import tensors_to_device

from model import load_best_model, make_model_and_optimizer
from asteroid import torch_utils
from train import FUSSSystem, ClipedNegSNR
from asteroid.data.fuss_dataset import FUSSDataset
from train import CustomLoss
from asteroid.losses.sdr import pairwise_neg_snr

parser = argparse.ArgumentParser()
parser.add_argument('--test_file', type=str, required=True,
                    help='Test txt file of FUSS')
parser.add_argument('--use_gpu', type=int, default=0,
                    help='Whether to use the GPU for model execution')
parser.add_argument('--exp_dir', default='exp/tmp',
                    help='Experiment root')
parser.add_argument('--n_save_ex', type=int, default=50,
                    help='Number of audio examples to save, -1 means all')

compute_metrics = ['si_sdr']


def load_model(train_conf, checpoint_path):
    """ Load best model after training.

    Args:
        train_conf (dict): dictionary as expected by `make_model_and_optimizer`
        exp_dir(str): Experiment directory. Expects to find
            `'best_k_models.json'` there.

    Returns:
        nn.Module the best pretrained model according to the val_loss.
    """
    # Create the model from recipe-local function
    model, _ = make_model_and_optimizer(train_conf)
    # Last best model summary
    #model = dummy(model)
    # Load checkpoint
    checkpoint = torch.load(checpoint_path, map_location='cpu')
    # Load state_dict into model.
    #model.load_state_dict(checkpoint["state_dict"], strict=True)
    model = torch_utils.load_state_dict_in(checkpoint['state_dict'],
                                           model)
    model.eval()
    return model




def main(conf):

    model = load_model(conf['train_conf'], os.path.join(conf['exp_dir'], "checkpoints", "_ckpt_epoch_3.ckpt")) #this raises an erro IDK why
    #model = FUSSSystem(conf["train_conf"])

    # Handle device placement
    if conf['use_gpu']:
        model.cuda()
    model_device = next(model.parameters()).device
    test_set = FUSSDataset(conf["test_file"], return_bg=True) #WhamDataset(conf['test_dir'], conf['task'],
                           #sample_rate=conf['sample_rate'],
                           #nondefault_nsrc=model.masker.n_src,
                           #segment=None)  # Uses all segment length

    # Used to reorder sources only
    #loss_func = PITLossWrapper(pairwise_neg_sisdr, mode='pairwise')
    loss_func = PITLossWrapper(pairwise_neg_sisdr)

    # Randomly choose the indexes of sentences to save.
    ex_save_dir = os.path.join(conf['exp_dir'], 'examples/')
    if conf['n_save_ex'] == -1:
        conf['n_save_ex'] = len(test_set)
    save_idx = random.sample(range(len(test_set)), conf['n_save_ex'])
    series_list = []
    torch.no_grad().__enter__()
    for idx in tqdm(range(len(test_set))):
        # Forward the network on the mixture.
        mix, sources, bg = tensors_to_device(test_set[idx], device=model_device)
        est_sources = model(mix[None, None])
        sources = sources.unsqueeze(0)
        _, reordered_sources = loss_func(est_sources, sources,
                                            return_est=True)

        silent = (torch.norm(sources, dim=-1, p=2) < 1e-8)
        if (silent).all():
            continue
        if (~silent).all():
            reordered_sources = reordered_sources.squeeze(0)
            sources = sources.squeeze(0)
        else:
            reordered_sources = reordered_sources[~silent]
            sources = sources[~silent]
        mix_np = mix[None].cpu().data.numpy()
        sources_np = sources.cpu().data.numpy()
        est_sources_np = reordered_sources.cpu().data.numpy()
        utt_metrics = get_metrics(mix_np, sources_np, est_sources_np,
                                  sample_rate=conf["sample_rate"], metrics_list=compute_metrics)
        utt_metrics['mix_path'] = test_set.mixtures[idx][0]
        series_list.append(pd.Series(utt_metrics))

        # Save some examples in a folder. Wav files and metrics as text.
        if idx in save_idx:
            local_save_dir = os.path.join(ex_save_dir, 'ex_{}/'.format(idx))
            os.makedirs(local_save_dir, exist_ok=True)
            sf.write(local_save_dir + "mixture.wav", mix_np[0],
                     conf['sample_rate'])
            # Loop over the sources and estimates
            for src_idx, src in enumerate(sources_np):
                sf.write(local_save_dir + "s{}.wav".format(src_idx+1), src,
                         conf['sample_rate'])
            for src_idx, est_src in enumerate(est_sources_np):
                sf.write(local_save_dir + "s{}_estimate.wav".format(src_idx+1),
                         est_src, conf['sample_rate'])
            # Write local metrics to the example folder.
            with open(local_save_dir + 'metrics.json', 'w') as f:
                json.dump(utt_metrics, f, indent=0)

    # Save all metrics to the experiment folder.
    all_metrics_df = pd.DataFrame(series_list)
    all_metrics_df.to_csv(os.path.join(conf['exp_dir'], 'all_metrics.csv'))

    # Print and save summary metrics
    final_results = {}
    for metric_name in compute_metrics:
        input_metric_name = 'input_' + metric_name
        ldf = all_metrics_df[metric_name] - all_metrics_df[input_metric_name]
        final_results[metric_name] = all_metrics_df[metric_name].mean()
        final_results[metric_name + '_imp'] = ldf.mean()
    print('Overall metrics :')
    pprint(final_results)
    with open(os.path.join(conf['exp_dir'], 'final_metrics.json'), 'w') as f:
        json.dump(final_results, f, indent=0)


if __name__ == '__main__':
    args = parser.parse_args()
    arg_dic = dict(vars(args))

    # Load training config
    conf_path = os.path.join(args.exp_dir, 'conf.yml')
    with open(conf_path) as f:
        train_conf = yaml.safe_load(f)
    arg_dic['sample_rate'] = 16000
    arg_dic['train_conf'] = train_conf

    #if args.task != arg_dic['train_conf']['data']['task']:
     #   print("Warning : the task used to test is different than "
      #        "the one from training, be sure this is what you want.")

    main(arg_dic)
