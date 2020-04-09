import os
from asteroid.masknn.blocks import TwoStepTDCN
from asteroid.engine.optimizers import make_optimizer
from asteroid import torch_utils

import torch
import torch.nn as nn


import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveEncoder1D(nn.Module):
    """
        A 1D convolutional block that transforms signal in wave form into higher
        dimension.

        Args:
            input shape: [batch, 1, n_samples]
            output shape: [batch, freq_res, n_samples//sample_res]
            freq_res: number of output frequencies for the encoding convolution
            sample_res: int, length of the encoding filter
    """

    def __init__(self, freq_res, sample_res):
        super().__init__()
        self.conv = nn.Conv1d(1,
                              freq_res,
                              sample_res,
                              stride=sample_res // 2,
                              padding=sample_res // 2)

    def forward(self, s):
        return F.relu(self.conv(s))


class AdaptiveDecoder1D(nn.Module):
    """ A 1D deconvolutional block that transforms encoded representation
    into wave form.
    input shape: [batch, freq_res, sample_res]
    output shape: [batch, 1, sample_res*n_samples]
    freq_res: number of output frequencies for the encoding convolution
    sample_res: length of the encoding filter
    """

    def __init__(self, freq_res, sample_res, n_sources):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(n_sources * freq_res,
                                         n_sources,
                                         sample_res,
                                         padding=sample_res // 2,
                                         stride=sample_res // 2,
                                         groups=n_sources,
                                         output_padding=(sample_res // 2) - 1)

    def forward(self, x):
        return self.deconv(x)


class AdaptiveEncoderDecoder(nn.Module):
    """
        Adaptive basis encoder and decoder with inference of ideal masks.
        Copied from: https://github.com/etzinis/two_step_mask_learning/

        Args:
            freq_res: The number of frequency like representations
            sample_res: The number of samples in kernel 1D convolutions
            n_sources: The number of sources
        References:
            Tzinis, E., Venkataramani, S., Wang, Z., Subakan, Y. C., and
            Smaragdis, P., "Two-Step Sound Source Separation:
            Training on Learned Latent Targets." In Acoustics, Speech
            and Signal Processing (ICASSP), 2020 IEEE International Conference.
            https://arxiv.org/abs/1910.09804
    """

    def __init__(self,
                 freq_res=256,
                 sample_res=21,
                 n_sources=2):
        super().__init__()
        self.freq_res = freq_res
        self.sample_res = sample_res
        self.mix_encoder = AdaptiveEncoder1D(freq_res, sample_res)
        self.decoder = AdaptiveDecoder1D(freq_res, sample_res, n_sources)
        self.n_sources = n_sources

    def get_target_masks(self, clean_sources):
        """
        Get target masks for the given clean sources
        :param clean_sources: [batch, n_sources, time_samples]
        :return: Ideal masks for the given sources:
        [batch, n_sources, time_samples//(sample_res // 2)]
        """
        enc_mask_list = [self.mix_encoder(clean_sources[:, i, :].unsqueeze(1))
                         for i in range(self.n_sources)]
        total_mask = torch.stack(enc_mask_list, dim=1)
        return F.softmax(total_mask, dim=1)

    def reconstruct(self, mixture):
        enc_mixture = self.mix_encoder(mixture.unsqueeze(1))
        return self.decoder(enc_mixture)

    def get_encoded_sources(self, mixture, clean_sources):
        enc_mixture = self.mix_encoder(mixture.unsqueeze(1))
        enc_masks = self.get_target_masks(clean_sources)
        s_recon_enc = enc_masks * enc_mixture.unsqueeze(1)
        return s_recon_enc

    def forward(self, mixture, clean_sources):
        enc_mixture = self.mix_encoder(mixture.unsqueeze(1))
        enc_masks = self.get_target_masks(clean_sources)

        s_recon_enc = enc_masks * enc_mixture.unsqueeze(1)
        recon_sources = self.decoder(s_recon_enc.view(s_recon_enc.shape[0],
                                                      -1,
                                                      s_recon_enc.shape[-1]))
        return recon_sources, enc_masks


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
        raise FileNotFoundError(
            'There are no available filterbanks under: {}\n'
            'Consider training one using train.py.'.format(f_checkpoint_dir)
        )

    exp_dir, checkpoint_dir = get_encoded_paths(conf, train_part='separator')
    model_available = False
    if os.path.exists(checkpoint_dir):
        available_models = os.listdir(checkpoint_dir)
        if available_models:
            model_available = True

    if not model_available:
        raise FileNotFoundError('There is no available separator model at: {}'
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
        model = AdaptiveEncoderDecoder(
            freq_res=conf['filterbank']['n_filters'],
            sample_res=conf['filterbank']['kernel_size'],
            n_sources=conf['masknet']['n_src'])
    elif model_part == 'separator':
        if pretrained_filterbank is None:
            raise ValueError('A pretrained filterbank is required for the '
                             'initialization of the separator.')
        model = Model(pretrained_filterbank, conf)
    else:
        raise ValueError('Part to train: {} is not available.'.format(
            model_part))
    # Define optimizer of this model
    optimizer = make_optimizer(
        model.parameters(),
        optimizer=conf[model_part + '_training'][model_part[0] + '_optimizer'],
        lr=conf[model_part + '_training'][model_part[0] + '_lr'])
    return model, optimizer
