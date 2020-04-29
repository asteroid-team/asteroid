import os
import json
import torch
from torch import nn
from asteroid.masknn import norms, activations, consistency
from asteroid.masknn.blocks import Conv1DBlock, TDConvNet
from asteroid.torch_utils import pad_x_to_y
from asteroid.utils import has_arg

import asteroid.filterbanks as fb
from asteroid import torch_utils
from asteroid.engine.optimizers import make_optimizer
from asteroid.filterbanks import transforms


class TDCNpp(nn.Module):
    """ Improved Temporal Convolutional network used in [1] (TDCN++)

    Args:
        in_chan (int): Number of input filters.
        n_src (int): Number of masks to estimate.
        out_chan (int, optional): Number of bins in the estimated masks.
            If ``None``, `out_chan = in_chan`.
        n_blocks (int, optional): Number of convolutional blocks in each
            repeat. Defaults to 8.
        n_repeats (int, optional): Number of repeats. Defaults to 3.
        bn_chan (int, optional): Number of channels after the bottleneck.
        hid_chan (int, optional): Number of channels in the convolutional
            blocks.
        skip_chan (int, optional): Number of channels in the skip connections.
            If 0 or None, TDConvNet won't have any skip connections and the
            masks will be computed from the residual output.
            Corresponds to the ConvTasnet architecture in v1 or the paper.
        kernel_size (int, optional): Kernel size in convolutional blocks.
        norm_type (str, optional): To choose from ``'BN'``, ``'gLN'``,
            ``'cLN'``.
        mask_act (str, optional): Which non-linear function to generate mask.

    References:
        [1] : Kavalerov, Ilya et al. “Universal Sound Separation.”
            in WASPAA 2019

    Notes:
        The differences wrt to ConvTasnet's TCN are
        1. Channel wise layer norm instead of global
        2. Longer-range skip-residual connections from earlier repeat inputs
            to later repeat inputs after passing them through dense layer.
        3. Learnable scaling parameter after each dense layer. The scaling
            parameter for the second dense  layer  in  each  convolutional
            block (which  is  applied  rightbefore the residual connection) is
            initialized to an exponentially decaying scalar equal to 0.9L,
            where L is the layer or block index.
    """
    def __init__(self, in_chan, n_src, out_chan=None, n_blocks=8, n_repeats=3,
                 bn_chan=128, hid_chan=512, skip_chan=128, kernel_size=3,
                 norm_type="cLN", mask_act='relu'):
        super().__init__()
        self.in_chan = in_chan
        self.n_src = n_src
        out_chan = out_chan if out_chan else in_chan
        self.out_chan = out_chan
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.bn_chan = bn_chan
        self.hid_chan = hid_chan
        self.skip_chan = skip_chan
        self.kernel_size = kernel_size
        self.norm_type = norm_type
        self.mask_act = mask_act

        layer_norm = norms.get(norm_type)(in_chan)
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)
        # Succession of Conv1DBlock with exponentially increasing dilation.
        self.TCN = nn.ModuleList()
        for r in range(n_repeats):
            for x in range(n_blocks):
                padding = (kernel_size - 1) * 2**x // 2
                self.TCN.append(Conv1DBlock(bn_chan, hid_chan, skip_chan,
                                            kernel_size, padding=padding,
                                            dilation=2**x, norm_type=norm_type))
        # Dense connection in TDCNpp
        self.dense_skip = nn.ModuleList()
        for r in range(n_repeats-1):
            self.dense_skip.append(nn.Conv1d(bn_chan, bn_chan, 1))

        scaling_param = torch.tensor([0.9**l for l in range(1, n_blocks)])
        scaling_param = scaling_param.unsqueeze(0).expand(n_repeats, n_blocks-1)
        self.scaling_param = nn.Parameter(scaling_param, requires_grad=True)

        mask_conv_inp = skip_chan if skip_chan else bn_chan
        mask_conv = nn.Conv1d(mask_conv_inp, n_src*out_chan, 1)
        self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)
        # Get activation function.
        mask_nl_class = activations.get(mask_act)
        # For softmax, feed the source dimension.
        if has_arg(mask_nl_class, 'dim'):
            self.output_act = mask_nl_class(dim=1)
        else:
            self.output_act = mask_nl_class()

    def forward(self, mixture_w):
        """

        Args:
            mixture_w (:class:`torch.Tensor`): Tensor of shape
                [batch, n_filters, n_frames]

        Returns:
            :class:`torch.Tensor`:
                estimated mask of shape [batch, n_src, n_filters, n_frames]
        """
        batch, n_filters, n_frames = mixture_w.size()
        output = self.bottleneck(mixture_w)
        output_copy = output

        skip_connection = 0.
        for r in range(self.n_repeats):
            # Long range skip connection TDCNpp
            if r != 0:
                # Transform the input to repeat r-1 and add to new repeat inp
                output = self.dense_skip[r-1](output_copy) + output
                # Copy this for later.
                output_copy = output
            for x in range(self.n_blocks):
                # Common to w. skip and w.o skip architectures
                i = r * self.n_blocks + x
                tcn_out = self.TCN[i](output)
                if self.skip_chan:
                    residual, skip = tcn_out
                    skip_connection = skip_connection + skip
                else:
                    residual = tcn_out
                # Initialized exp decay scale factor TDCNpp
                scale = self.scaling_param[r, x-1] if x > 0 else 1.
                residual = residual * scale
                output = output + residual
        # Use residual output when no skip connection
        mask_inp = skip_connection if self.skip_chan else output
        score = self.mask_net(mask_inp)
        score = score.view(batch, self.n_src, self.out_chan, n_frames)
        est_mask = self.output_act(score)
        return est_mask


class Model(nn.Module):
    def __init__(self, encoder, masker, decoder):
        super().__init__()
        self.encoder = encoder
        self.masker = masker
        self.decoder = decoder

    def forward(self, x, bg=None):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        tf_rep = self.encoder(x)
        # Concat ReIm and Mag input
        est_masks = self.masker(transforms.take_cat(tf_rep))
        # Note : this is equivalent to ReIm masking for STFT
        masked_tf_rep = est_masks * tf_rep.unsqueeze(1)
        out_wavs = self.decoder(masked_tf_rep)
        # Mixture consistency (weights are not learned but based on power)
        # Estimates should sum up to the targets only
        if bg is None:
            return pad_x_to_y(out_wavs, x)
        if len(bg.shape) == 2:
            bg = bg.unsqueeze(1)
        out_wavs = consistency.mixture_consistency(x - bg, out_wavs)
        return pad_x_to_y(out_wavs, x)


def make_model_and_optimizer(conf):
    """ Function to define the model and optimizer for a config dictionary.
    Args:
        conf: Dictionary containing the output of hierachical argparse.
    Returns:
        model, optimizer.
    The main goal of this function is to make reloading for resuming
    and evaluation very simple.
    """
    # Define building blocks for local model
    # Filterbank can be either stftfb or freefb
    enc, dec = fb.make_enc_dec(**conf['filterbank'])
    mask_conf = dict(conf['masknet'])  # Make a copy
    improved = mask_conf.pop('improved')
    # We will take magnitude and concat with ReIm
    if improved:
        masker = TDCNpp(in_chan=3 * enc.filterbank.n_feats_out // 2,
                        out_chan=enc.filterbank.n_feats_out,
                        n_src=3,  # Hardcoded here because of FUSS
                        **mask_conf)
    else:
        masker = TDConvNet(in_chan=3 * enc.filterbank.n_feats_out // 2,
                           out_chan=enc.filterbank.n_feats_out,
                           n_src=3,  # Hardcoded here because of FUSS
                           **mask_conf)

    model = Model(enc, masker, dec)
    # Define optimizer of this model
    optimizer = make_optimizer(model.parameters(), **conf['optim'])
    return model, optimizer


def load_best_model(train_conf, exp_dir):
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
    with open(os.path.join(exp_dir, 'best_k_models.json'), "r") as f:
        best_k = json.load(f)
    best_model_path = min(best_k, key=best_k.get)
    # Load checkpoint
    checkpoint = torch.load(best_model_path, map_location='cpu')
    # Load state_dict into model.
    model = torch_utils.load_state_dict_in(checkpoint['state_dict'],
                                           model)
    model.eval()
    return model
