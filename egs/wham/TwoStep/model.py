import os
from asteroid.engine.optimizers import make_optimizer
from asteroid import torch_utils

import torch
import torch.nn as nn


import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoStepTDCN(nn.Module):
    """
    A time-dilated convolutional network (TDCN) similar to the initial
    ConvTasNet architecture where the encoder and decoder have been
    pre-trained separately. The TwoStepTDCN infers masks directly on the
    latent space and works using an signal to distortion ratio (SDR) loss
    directly on the ideal latent masks.
    Adaptive basis encoder and decoder with inference of ideal masks.
    Copied from: https://github.com/etzinis/two_step_mask_learning/

    Args:
        pretrained_filterbank: A pretrained encoder decoder like the one
            implemented in asteroid_filterbanks.simple_adaptive
        n_sources (int, optional): Number of masks to estimate.
        n_blocks (int, optional): Number of convolutional blocks in each
            repeat. Defaults to 8.
        n_repeats (int, optional): Number of repeats. Defaults to 4.
        bn_chan (int, optional): Number of channels after the bottleneck.
        hid_chan (int, optional): Number of channels in the convolutional
            blocks.
        kernel_size (int, optional): Kernel size in convolutional blocks.
            n_sources: The number of sources
    References:
        Tzinis, E., Venkataramani, S., Wang, Z., Subakan, Y. C., and
        Smaragdis, P., "Two-Step Sound Source Separation:
        Training on Learned Latent Targets." In Acoustics, Speech
        and Signal Processing (ICASSP), 2020 IEEE International Conference.
        https://arxiv.org/abs/1910.09804
    """

    def __init__(
        self,
        pretrained_filterbank,
        bn_chan=256,
        hid_chan=512,
        kernel_size=3,
        n_blocks=8,
        n_repeats=4,
        n_sources=2,
    ):
        super(TwoStepTDCN, self).__init__()
        try:
            self.pretrained_filterbank = pretrained_filterbank
            self.encoder = self.pretrained_filterbank.mix_encoder
            self.decoder = self.pretrained_filterbank.decoder
            self.fbank_basis = self.encoder.conv.out_channels
            self.fbank_kernel_size = self.encoder.conv.kernel_size[0]

            # Freeze the encoder and the decoder weights:
            self.encoder.conv.weight.requires_grad = False
            self.encoder.conv.bias.requires_grad = False
            self.decoder.deconv.weight.requires_grad = False
            self.decoder.deconv.bias.requires_grad = False
        except Exception as e:
            print(e)
            raise ValueError("Could not load features form the pretrained " "adaptive filterbank.")

        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.bn_chan = bn_chan
        self.hid_chan = hid_chan
        self.kernel_size = kernel_size
        self.n_sources = n_sources

        # Norm before the rest, and apply one more dense layer
        self.ln_in = nn.BatchNorm1d(self.fbank_basis)
        # self.ln_in = GlobalLayerNorm(self.fbank_basis)
        self.l1 = nn.Conv1d(in_channels=self.fbank_basis, out_channels=self.bn_chan, kernel_size=1)

        # Separation module
        self.separator = nn.Sequential(
            *[
                SeparableDilatedConv1DBlock(
                    in_chan=self.bn_chan,
                    hid_chan=self.hid_chan,
                    kernel_size=self.kernel_size,
                    dilation=2**d,
                )
                for _ in range(self.n_blocks)
                for d in range(self.n_repeats)
            ]
        )

        # Masks layer
        self.mask_layer = nn.Conv2d(
            in_channels=1,
            out_channels=self.n_sources,
            kernel_size=(self.fbank_basis + 1, 1),
            padding=(self.fbank_basis - self.fbank_basis // 2, 0),
        )

        # Reshaping if needed
        if self.bn_chan != self.fbank_basis:
            self.out_reshape = nn.Conv1d(
                in_channels=self.bn_chan, out_channels=self.fbank_basis, kernel_size=1
            )
        self.ln_mask_in = nn.BatchNorm1d(self.fbank_basis)
        # self.ln_mask_in = GlobalLayerNorm(self.fbank_basis)

    def forward(self, x):
        # Front end
        x = self.encoder(x)
        encoded_mixture = x.clone()

        # Separation module
        x = self.ln_in(x)
        x = self.l1(x)
        x = self.separator(x)

        if self.bn_chan != self.fbank_basis:
            x = self.out_reshape(x)

        x = self.ln_mask_in(x)
        x = nn.functional.relu(x)
        x = self.mask_layer(x.unsqueeze(1))
        masks = nn.functional.softmax(x, dim=1)
        return masks * encoded_mixture.unsqueeze(1)

    def infer_source_signals(self, mixture_wav):
        adfe_sources = self.forward(mixture_wav)
        rec_wavs = self.decoder(
            adfe_sources.view(adfe_sources.shape[0], -1, adfe_sources.shape[-1])
        )
        return rec_wavs


class SeparableDilatedConv1DBlock(nn.Module):
    """One dimensional convolutional block, as proposed in [1] without skip
        output. As used in the two step approach [2]. This block uses the
        groupnorm across features and also produces always a padded output.

    Args:
        in_chan (int): Number of input channels.
        hid_chan (int): Number of hidden channels in the depth-wise
            convolution.
        kernel_size (int): Size of the depth-wise convolutional kernel.
        dilation (int): Dilation of the depth-wise convolution.

    References:
        [1]: "Conv-TasNet: Surpassing ideal time-frequency magnitude masking
             for speech separation" TASLP 2019 Yi Luo, Nima Mesgarani
             https://arxiv.org/abs/1809.07454
        [2]: Tzinis, E., Venkataramani, S., Wang, Z., Subakan, Y. C., and
            Smaragdis, P., "Two-Step Sound Source Separation:
            Training on Learned Latent Targets." In Acoustics, Speech
            and Signal Processing (ICASSP), 2020 IEEE International Conference.
            https://arxiv.org/abs/1910.09804
    """

    def __init__(self, in_chan=256, hid_chan=512, kernel_size=3, dilation=1):
        super(SeparableDilatedConv1DBlock, self).__init__()
        self.module = nn.Sequential(
            nn.Conv1d(in_channels=in_chan, out_channels=hid_chan, kernel_size=1),
            nn.PReLU(),
            nn.GroupNorm(1, hid_chan, eps=1e-08),
            nn.Conv1d(
                in_channels=hid_chan,
                out_channels=hid_chan,
                kernel_size=kernel_size,
                padding=(dilation * (kernel_size - 1)) // 2,
                dilation=dilation,
                groups=hid_chan,
            ),
            nn.PReLU(),
            nn.GroupNorm(1, hid_chan, eps=1e-08),
            nn.Conv1d(in_channels=hid_chan, out_channels=in_chan, kernel_size=1),
        )

    def forward(self, x):
        """Input shape [batch, feats, seq]"""
        y = x.clone()
        return x + self.module(y)


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
        self.conv = nn.Conv1d(
            1, freq_res, sample_res, stride=sample_res // 2, padding=sample_res // 2
        )

    def forward(self, s):
        return F.relu(self.conv(s))


class AdaptiveDecoder1D(nn.Module):
    """A 1D deconvolutional block that transforms encoded representation
    into wave form.
    input shape: [batch, freq_res, sample_res]
    output shape: [batch, 1, sample_res*n_samples]
    freq_res: number of output frequencies for the encoding convolution
    sample_res: length of the encoding filter
    """

    def __init__(self, freq_res, sample_res, n_sources):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(
            n_sources * freq_res,
            n_sources,
            sample_res,
            padding=sample_res // 2,
            stride=sample_res // 2,
            groups=n_sources,
            output_padding=(sample_res // 2) - 1,
        )

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

    def __init__(self, freq_res=256, sample_res=21, n_sources=2):
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
        enc_mask_list = [
            self.mix_encoder(clean_sources[:, i, :].unsqueeze(1)) for i in range(self.n_sources)
        ]
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
        recon_sources = self.decoder(
            s_recon_enc.view(s_recon_enc.shape[0], -1, s_recon_enc.shape[-1])
        )
        return recon_sources, enc_masks


class Model(nn.Module):
    def __init__(self, pretrained_filterbank, conf):
        super().__init__()
        self.pretrained_filterbank = pretrained_filterbank
        self.separator = TwoStepTDCN(
            pretrained_filterbank,
            bn_chan=conf["masknet"]["bn_chan"],
            hid_chan=conf["masknet"]["hid_chan"],
            kernel_size=conf["masknet"]["conv_kernel_size"],
            n_blocks=conf["masknet"]["n_blocks"],
            n_repeats=conf["masknet"]["n_repeats"],
            n_sources=conf["masknet"]["n_src"],
        )

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
        return self.pretrained_filterbank.get_encoded_sources(mixture, clean_sources)

    def estimate_latent_representations(self, mixture):
        return self.separator(mixture.unsqueeze(1))

    def get_ideal_latent_targets(self, mixture, clean_sources):
        return self.pretrained_filterbank.get_encoded_sources(mixture, clean_sources)

    def forward(self, x):
        # Infer sources in the time domain
        return self.separator.infer_source_signals(x.unsqueeze(1))


def get_encoded_paths(conf, train_part=None):
    exp_dir = conf["main_args"]["exp_dir"]
    N = conf["filterbank"]["n_filters"]
    L = conf["filterbank"]["kernel_size"]
    checkpoint_dir = os.path.join(exp_dir, train_part, "N_{}_L_{}_checkpoints/".format(N, L))
    return exp_dir, checkpoint_dir


def load_best_separator_if_available(conf):
    filterbank = load_best_filterbank_if_available(conf)
    _, f_checkpoint_dir = get_encoded_paths(conf, "filterbank")
    if filterbank is None:
        raise FileNotFoundError(
            "There are no available filterbanks under: {}\n"
            "Consider training one using train.py.".format(f_checkpoint_dir)
        )

    exp_dir, checkpoint_dir = get_encoded_paths(conf, train_part="separator")
    model_available = False
    if os.path.exists(checkpoint_dir):
        available_models = [p for p in os.listdir(checkpoint_dir) if ".ckpt" in p]
        if available_models:
            model_available = True

    if not model_available:
        raise FileNotFoundError(
            "There is no available separator model at: {}" "".format(checkpoint_dir)
        )

    model_path = os.path.join(checkpoint_dir, available_models[0])
    print("Going to load from: {}".format(model_path))
    checkpoint = torch.load(model_path, map_location="cpu")
    model_c, _ = make_model_and_optimizer(
        conf, model_part="separator", pretrained_filterbank=filterbank
    )
    model = torch_utils.load_state_dict_in(checkpoint["state_dict"], model_c)
    print("Successfully loaded separator from: {}".format(model_path))
    return model


def load_best_filterbank_if_available(conf):
    exp_dir, checkpoint_dir = get_encoded_paths(conf, train_part="filterbank")

    filterbank_available = False
    if os.path.exists(checkpoint_dir):
        available_filter_banks = [p for p in os.listdir(checkpoint_dir) if ".ckpt" in p]
        if available_filter_banks:
            filterbank_available = True

    if not filterbank_available:
        return None

    filterbank_path = os.path.join(checkpoint_dir, available_filter_banks[0])
    print("Going to load from: {}".format(filterbank_path))
    checkpoint = torch.load(filterbank_path, map_location="cpu")
    # Update number of source values (It depends on the task)
    conf["masknet"].update({"n_src": checkpoint["training_config"]["masknet"]["n_src"]})
    filterbank, _ = make_model_and_optimizer(conf, model_part="filterbank")
    model = torch_utils.load_state_dict_in(checkpoint["state_dict"], filterbank)
    print("Successfully loaded filterbank from: {}".format(filterbank_path))
    return model


def make_model_and_optimizer(conf, model_part="filterbank", pretrained_filterbank=None):
    """Function to define the model and optimizer for a config dictionary.
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
    if model_part == "filterbank":
        model = AdaptiveEncoderDecoder(
            freq_res=conf["filterbank"]["n_filters"],
            sample_res=conf["filterbank"]["kernel_size"],
            n_sources=conf["masknet"]["n_src"],
        )
    elif model_part == "separator":
        if pretrained_filterbank is None:
            raise ValueError(
                "A pretrained filterbank is required for the " "initialization of the separator."
            )
        model = Model(pretrained_filterbank, conf)
    else:
        raise ValueError("Part to train: {} is not available.".format(model_part))
    # Define optimizer of this model
    optimizer = make_optimizer(
        model.parameters(),
        optimizer=conf[model_part + "_training"][model_part[0] + "_optimizer"],
        lr=conf[model_part + "_training"][model_part[0] + "_lr"],
    )
    return model, optimizer
