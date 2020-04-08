import argparse
import json
import os
import random
from pprint import pprint
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import yaml
from pb_bss.evaluation import InputMetrics, OutputMetrics
from tqdm import tqdm

from asteroid.data.librimix_dataset import LibriMix
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid.torch_utils import load_state_dict_in
from asteroid.utils import tensors_to_device, average_arrays_in_dic
from model import make_model_and_optimizer

parser = argparse.ArgumentParser()
parser.add_argument('--test_dir', type=str, required=True,
                    help='Test directory including the json files')
parser.add_argument('--use_gpu', type=int, default=0,
                    help='Whether to use the GPU for model execution')
parser.add_argument('--exp_dir', default='exp/tmp',
                    help='Experiment root')
parser.add_argument('--n_save_ex', type=int, default=10,
                    help='Number of audio examples to save, -1 means all')

compute_metrics = ['si_sdr', 'mir_eval_sdr', 'mir_eval_sir', 'mir_eval_sar',
                   'stoi']


def main(conf):
    # Make the model
    model, _ = make_model_and_optimizer(conf['train_conf'])
    # Load best model
    with open(os.path.join(conf['exp_dir'], 'best_k_models.json'), "r") as f:
        best_k = json.load(f)
    best_model_path = min(best_k, key=best_k.get)
    # Load checkpoint
    checkpoint = torch.load(best_model_path, map_location='cpu')
    state = checkpoint['state_dict']
    state_copy = state.copy()
    # Remove unwanted keys
    for keys, values in state.items():
        if keys.startswith('loss'):
            del state_copy[keys]
            print(keys)
    model = load_state_dict_in(state_copy, model)

    # Handle device placement
    if conf['use_gpu']:
        model.cuda()
    model_device = next(model.parameters()).device

    test_set = LibriMix(conf['test_dir'], None,
                        conf['sample_rate'],
                        conf['train_conf']['data']['n_src'])

    loss_func = PITLossWrapper(pairwise_neg_sisdr, mode='pairwise')

    # Randomly choose the indexes of sentences to save.
    ex_save_dir = os.path.join(conf['exp_dir'], 'examples_mss_8K/')
    if conf['n_save_ex'] == -1:
        conf['n_save_ex'] = len(test_set)
    save_idx = random.sample(range(len(test_set)), conf['n_save_ex'])
    series_list = []
    torch.no_grad().__enter__()
    for idx in tqdm(range(len(test_set))):
        # Forward the network on the mixture.
        mix, sources = tensors_to_device(test_set[idx], device=model_device)

        est_sources = model(mix.unsqueeze(0))

        loss, reordered_sources = loss_func(est_sources, sources[None],
                                            return_est=True)

        mix_np = mix.cpu().data.numpy()
        sources_np = sources.squeeze().cpu().data.numpy()
        est_sources_np = reordered_sources.squeeze().cpu().data.numpy()
        est_sources_np[0, :] = est_sources_np[0, :] / \
                               np.max(est_sources_np, axis=1)[0] * \
                               np.max(sources_np, axis=1)[0]
        est_sources_np[1, :] = est_sources_np[1, :] / \
                               np.max(est_sources_np, axis=1)[1] * \
                               np.max(sources_np, axis=1)[1]
        # For each utterance, we get a dictionary with the mixture path,
        # the input and output metrics.utt_metrics
        input_metrics = InputMetrics(observation=mix_np,
                                     speech_source=sources_np,
                                     enable_si_sdr=True,
                                     sample_rate=conf['sample_rate'])
        utt_metrics = {'input_' + n: input_metrics[n] for n in compute_metrics}

        output_metrics = OutputMetrics(speech_prediction=est_sources_np,
                                       speech_source=sources_np,
                                       enable_si_sdr=True,
                                       sample_rate=conf['sample_rate'])

        utt_metrics.update(output_metrics[compute_metrics])

        utt_metrics = average_arrays_in_dic(utt_metrics)
        series_list.append(pd.Series(utt_metrics))

        # Save some examples in a folder. Wav files and metrics as text.
        if idx in save_idx:
            local_save_dir = os.path.join(ex_save_dir, 'ex_{}/'.format(idx))
            os.makedirs(local_save_dir, exist_ok=True)
            sf.write(local_save_dir + "mixture.wav", mix_np[0],
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
    all_metrics_df = pd.DataFrame(series_list)
    all_metrics_df.to_csv(os.path.join(ex_save_dir, 'all_metrics.csv'))

    # Print and save summary metrics
    final_results = {}
    for metric_name in compute_metrics:
        input_metric_name = 'input_' + metric_name
        ldf = all_metrics_df[metric_name] - all_metrics_df[input_metric_name]
        final_results[metric_name] = all_metrics_df[metric_name].mean()
        final_results[metric_name + '_imp'] = ldf.mean()
    print('Overall metrics :')
    pprint(final_results)
    with open(os.path.join(ex_save_dir, 'final_metrics.json'), 'w') as f:
        json.dump(final_results, f, indent=0)


if __name__ == '__main__':
    args = parser.parse_args()
    arg_dic = dict(vars(args))
    # Load training config
    conf_path = os.path.join(args.exp_dir, 'conf.yml')
    with open(conf_path) as f:
        train_conf = yaml.safe_load(f)
    arg_dic['sample_rate'] = train_conf['data']['sample_rate']
    arg_dic['train_conf'] = train_conf
    main(arg_dic)
