import os
import random

import soundfile as sf
import torch
import yaml
import json
import argparse
import pandas as pd

from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from model import make_model_and_optimizer
from asteroid.data.wham_dataset import WhamDataset
from asteroid.utils import tensors_to_device
from pb_bss.evaluation import InputMetrics, OutputMetrics

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, required=True,
                    help='One of `enh_single`, `enh_both`, '
                         '`sep_clean` or `sep_noisy`')
parser.add_argument('--test_dir', type=str, required=True,
                    help='Test directory including the json files')
parser.add_argument('--sample_rate', type=int, default=8000,
                    help='Sampling frequency in Hz')
parser.add_argument('--nondefault_nsrc', type=int, default=None,
                    help='Different number of sources from the default task')
parser.add_argument('--use_gpu', type=int, default=0,
                    help='Whether to use the GPU for model execution')
parser.add_argument('--exp_dir', default='exp/tmp',
                    help='Experiment root')
parser.add_argument('--n_save_ex', type=int, default=50,
                    help='Number of denoising examples to save')


def get_model(conf):
    # Create the model from recipe-local function
    model, _ = make_model_and_optimizer(conf['train_conf'])
    # Last best model summary
    with open(os.path.join(conf['exp_dir'], 'best_k_models.json'), "r") as f:
        best_k = json.load(f)
    best_model_path = min(best_k, key=best_k.get)
    # Load checkpoint
    model.load_state_dict(torch.load(best_model_path, map_location='cpu'))
    # Handle device placement
    if conf['use_gpu']:
        model.cuda()
    model.eval()
    return model


def main(conf):
    model = get_model(conf)
    model_device = next(model.parameters()).device
    test_set = WhamDataset(conf['test_dir'], conf['task'],
                           sample_rate=conf['sample_rate'],
                           nondefault_nsrc=conf['nondefault_nsrc'],
                           segment=None)  # Uses all segment length
    # Used to reorder sources only
    loss_func = PITLossWrapper(pairwise_neg_sisdr, mode='pairwise')

    all_metrics_df = pd.DataFrame()
    # Randomly choose the indexes of sentences to save.
    ex_save_dir = os.path.join(conf['exp_dir'], 'examples/')
    if conf['n_save_ex'] == -1:
        conf['n_save_ex'] = len(test_set)
    save_idx = random.sample(range(len(test_set)), conf['n_save_ex'])
    for idx in range(len(test_set)):
        # Forward the network on the mixture.
        mix, sources, _ = tensors_to_device(test_set[idx], device=model_device)
        est_sources = model(mix)
        loss, reordered_sources = loss_func(sources, est_sources,
                                            return_est=True)
        mix_np = mix.data.numpy()[0]
        sources_np = sources.data.numpy()[0]
        est_sources_np = reordered_sources.data.numpy()[0]
        # For each utterance, we get a dictionary with the mixture path,
        # the input and output metrics.
        utt_metrics = {'mix_path': test_set.mix[idx][0]}

        input_metrics = InputMetrics(observation=mix_np,
                                     speech_source=sources_np,
                                     enable_si_sdr=True,
                                     sample_rate=conf['sample_rate'])
        utt_metrics.update({'input_' + n : input_metrics[n] for n in
                            input_metrics._available_metric_names()})

        output_metrics = OutputMetrics(speech_prediction=est_sources_np,
                                       speech_source=sources_np,
                                       enable_si_sdr=True,
                                       sample_rate=conf['sample_rate'])
        utt_metrics.update(output_metrics.as_dict())
        all_metrics_df.append(pd.DataFrame.from_dict(utt_metrics),
                              ignore_index=True)

        if idx in save_idx:
            local_save_dir = os.path.join(ex_save_dir, 'ex_{}/'.format(idx))
            os.makedirs(local_save_dir, exist_ok=True)
            sf.write(local_save_dir + "mixture.wav", mix_np,
                     conf['sample_rate'])
            # Loop over the sources and estimates
            for src_idx, src in enumerate(sources_np):
                sf.write(local_save_dir + "s{}.wav".format(src_idx), src,
                         conf['sample_rate'])
            for src_idx, est_src in enumerate(est_sources_np):
                sf.write(local_save_dir + "s{}_estimate.wav".format(src_idx),
                         est_src, conf['sample_rate'])
            # Write local metrics to the example folder.
            with open(local_save_dir + 'metrics.json', 'w') as f:
                json.dump(utt_metrics, f, indent=0)
    # Save all metrics to the experiment folder.
    all_metrics_df.to_csv(os.path.join(conf['exp_dir'], 'all_metrics.csv'),
                          sep='\t')


if __name__ == '__main__':
    args = parser.parse_args()

    # Load training config
    conf_path = os.path.join(args.exp_dir, 'conf.yml')
    with open(conf_path) as f:
        train_conf = yaml.safe_load(f)
    arg_dic = dict(vars(args))
    arg_dic['train_conf'] = train_conf

    main(arg_dic)
