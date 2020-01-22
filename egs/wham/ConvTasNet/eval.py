import os
import torch
import yaml
import argparse

from asteroid.engine.losses import pairwise_neg_sisdr, PITLossContainer
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
# Add argument for : number of ex to save?
# Add argument for : STOI and PESQ computation?


def get_model(conf):
    # Create the model from recipe-local function
    model, _ = make_model_and_optimizer(conf['train_conf'])
    # Load state dict
    last_model_path = os.path.join(conf['exp_dir'], 'final.pth')
    model.load_state_dict(torch.load(last_model_path,
                                     map_location='cpu'))
    # Handle device placement
    if conf['use_gpu']:
        model.cuda()
    model.eval()
    return model


def main(conf):
    model = get_model(conf)
    test_set = WhamDataset(conf['test_dir'], conf['task'],
                           sample_rate=conf['sample_rate'],
                           nondefault_nsrc=conf['nondefault_nsrc'],
                           segment=None)
    loss_func = PITLossWrapper(pairwise_neg_sisdr, mode='pairwise')
    model_device = next(model.parameters()).device
    for idx in range(len(test_set)):
        mix, sources, _ = tensors_to_device(test_set[idx], device=model_device)
        est_sources = model(mix)
        loss, reordered_sources = loss_func(sources, est_sources,
                                            return_est=True)
        mix_np = mix.data.numpy()[0]
        sources_np = sources.data.numpy()[0]
        est_sources_np = reordered_sources.data.numpy()[0]
        # Waiting for pb_bss support to compute subset of metrics.
        # We will probably want SI-SDR,  + add option for mir_eval SDR, stoi,
        # pesq
        input_metrics = InputMetrics(observation=mix_np,
                                     speech_source=sources_np,
                                     enable_si_sdr=True,
                                     sample_rate=conf["sample_rate"])
        output_metrics = OutputMetrics(speech_prediction=est_sources_np,
                                       speech_source=sources_np,
                                       enable_si_sdr=True,
                                       sample_rate=conf["sample_rate"])


""" Left to do
- Compute the metrics and log them
- Save some examples to listen to 
- Write all metrics to files in the right folder.
"""


if __name__ == '__main__':
    args = parser.parse_args()

    # Load training config
    conf_path = os.path.join(args.exp_dir, 'conf.yml')
    with open(conf_path) as f:
        train_conf = yaml.safe_load(f)
    arg_dic = dict(vars(args))
    arg_dic['train_conf'] = train_conf

    main(arg_dic)
