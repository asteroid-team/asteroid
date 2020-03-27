import os
import asteroid.filterbanks.simple_adaptive as fb
from asteroid.masknn.blocks import TwoStepTDCN
from asteroid.engine.optimizers import make_optimizer
from asteroid import torch_utils

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, pretrained_filterbank, conf):
        super().__init__()
        self.pretrained_filterbank = pretrained_filterbank
        self.separator = TwoStepTDCN(
            pretrained_filterbank,
            bn_chan=conf['masknet']['bn_chan'],
            hid_chan=conf['masknet']['hid_chan'],
            kernel_size=conf['masknet']['conv_kernel_size'],
            n_blocks=conf['masknet']['n_blocks'],
            n_repeats=conf['masknet']['n_repeats'],
            n_sources=conf['masknet']['n_src'])

    def get_ideal_targets(self, mixture, clean_sources):
        """
        Get the latent targets for all sources
        :param mixture: Input mixture in time domain [batch, timesamples]
        :param clean_sources: Clean sources that constitute to the mixture in
            time domain [batch, n_sources, timesamples].
        :return: Latent representations for the sources which can be used as
            targets for training:
            [batch, n_sources, timesamples//encoder_stride]
        """
        return self.pretrained_filterbank.get_encoded_sources(mixture,
                                                              clean_sources)

    def estimate_latent_representations(self, mixture):
        return self.separator(mixture.unsqueeze(1))

    def get_ideal_latent_targets(self, mixture, clean_sources):
        return self.pretrained_filterbank.get_encoded_sources(mixture,
                                                              clean_sources)

    def forward(self, x):
        # Infer sources in the time domain
        return self.separator.infer_source_signals(x.unsqueeze(1))


def get_encoded_paths(conf, train_part=None):
    exp_dir = conf['main_args']['exp_dir']
    N = conf['filterbank']['n_filters']
    L = conf['filterbank']['kernel_size']
    checkpoint_dir = os.path.join(
        exp_dir, train_part, 'N_{}_L_{}_checkpoints/'.format(N, L))
    return exp_dir, checkpoint_dir


def load_best_separator_if_available(conf):
    filterbank = load_best_filterbank_if_available(conf)
    _, f_checkpoint_dir = get_encoded_paths(conf, 'filterbank')
    if filterbank is None:
        raise ImportError('There are no available filterbanks under: {}\n'
              'Consider training one using train.py.'.format(f_checkpoint_dir))

    exp_dir, checkpoint_dir = get_encoded_paths(conf, train_part='separator')
    model_available = False
    if os.path.exists(checkpoint_dir):
        available_models = os.listdir(checkpoint_dir)
        if available_models:
            model_available = True

    if not model_available:
        raise ImportError('There is no available separator model at: {}'
                          ''.format(checkpoint_dir))

    model_path = os.path.join(checkpoint_dir, available_models[0])
    checkpoint = torch.load(model_path, map_location='cpu')
    model_c, _ = make_model_and_optimizer(conf, model_part='separator',
                                          pretrained_filterbank=filterbank)
    model = torch_utils.load_state_dict_in(checkpoint['state_dict'], model_c)
    print('Successfully loaded separator from: {}'.format(model_path))
    return model


def load_best_filterbank_if_available(conf):
    exp_dir, checkpoint_dir = get_encoded_paths(conf, train_part='filterbank')

    filterbank_available = False
    if os.path.exists(checkpoint_dir):
        available_filter_banks = os.listdir(checkpoint_dir)
        if available_filter_banks:
           filterbank_available = True

    if not filterbank_available:
        return None

    filterbank_path = os.path.join(checkpoint_dir, available_filter_banks[0])
    checkpoint = torch.load(filterbank_path, map_location='cpu')
    # Update number of source values (It depends on the task)
    conf['masknet'].update(
        {'n_src': checkpoint['training_config']['masknet']['n_src']})
    filterbank, _ = make_model_and_optimizer(conf, model_part='filterbank')
    model = torch_utils.load_state_dict_in(checkpoint['state_dict'], filterbank)
    print('Successfully loaded filterbank from: {}'.format(filterbank_path))
    return model


def make_model_and_optimizer(conf, model_part='filterbank',
                             pretrained_filterbank=None):
    """ Function to define the model and optimizer for a config dictionary.
    Args:
        conf: Dictionary containing the output of hierachical argparse.
        model_part: Either filterbank (in other words adaptive front-end and
        back-end) or separator.
        pretrained_filterbank: The separator needs a pre-trained filterbank
            in order to be initialized appropriately.
    Returns:
        model, optimizer.
    The main goal of this function is to make reloading for resuming
    and evaluation very simple.
    """
    # Define building blocks for local model
    if model_part == 'filterbank':
        model = fb.AdaptiveEncoderDecoder(
            freq_res=conf['filterbank']['n_filters'],
            sample_res=conf['filterbank']['kernel_size'],
            n_sources=conf['masknet']['n_src'])
    elif model_part == 'separator':
        if pretrained_filterbank is None:
            raise ValueError('A pretrained filterbank is required for the '
                             'initialization of the separator.')
        model = Model(pretrained_filterbank, conf)
    else:
        raise NotImplementedError('Part to train: {} is not '
                                  'available.'.format(model_part))
    # Define optimizer of this model
    optimizer = make_optimizer(
        model.parameters(),
        optimizer=conf[model_part + '_training'][model_part[0] + '_optimizer'],
        lr=conf[model_part + '_training'][model_part[0] + '_lr'])
    return model, optimizer
