from torch import nn
import torch
from asteroid.masknn import norms
from asteroid.masknn import activations
from asteroid.utils import has_arg


class Conv1DBlock(nn.Module):

    def __init__(self, hid_chan, kernel_size, padding,
                 dilation, norm_type="gLN"):
        super(Conv1DBlock, self).__init__()

        conv_norm = norms.get(norm_type)
        depth_conv1d = nn.Conv1d(hid_chan, hid_chan, kernel_size,
                                 padding=padding, dilation=dilation,
                                 groups=hid_chan)

        self.out = nn.Sequential(depth_conv1d, nn.PReLU(), conv_norm(hid_chan))


    def forward(self, x):
        """ Input shape [batch, feats, seq]"""

        return self.out(x)

class SepConv1DBlock(nn.Module):

    def __init__(self, in_chan_spk_vec, hid_chan, kernel_size, padding,
                 dilation, norm_type="gLN", use_FiLM=True):
        super(SepConv1DBlock, self).__init__()

        self.use_FiLM = use_FiLM
        conv_norm = norms.get(norm_type)
        self.depth_conv1d = nn.Conv1d(hid_chan, hid_chan, kernel_size,
                                 padding=padding, dilation=dilation,
                                 groups=hid_chan)
        self.out = nn.Sequential(nn.PReLU(),
                                          conv_norm(hid_chan))

        # FiLM conditioning
        if self.use_FiLM:
            self.mul_lin = nn.Linear(in_chan_spk_vec, hid_chan)
        self.add_lin = nn.Linear(in_chan_spk_vec, hid_chan)

    def apply_conditioning(self, spk_vec, squeezed):
        bias = self.add_lin(spk_vec.transpose(1, -1)).transpose(1, -1)
        if self.use_FiLM:
            mul = self.mul_lin(spk_vec.transpose(1, -1)).transpose(1, -1)
            return mul*squeezed + bias
        else:
            return squeezed + bias

    def forward(self, x, spk_vec):
        """ Input shape [batch, feats, seq]"""

        conditioned = self.apply_conditioning(spk_vec, self.depth_conv1d(x))

        return self.out(conditioned)


class SpeakerStack(nn.Module):
    # basically this is plain conv-tasnet remove this in future releases

    def __init__(self, in_chan, n_src, n_blocks=14, n_repeats=1,
                 kernel_size=3,
                 norm_type="gLN"):
        
        super(SpeakerStack, self).__init__()
        self.in_chan = in_chan
        self.n_src = n_src
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.kernel_size = kernel_size
        self.norm_type = norm_type

        #layer_norm = norms.get(norm_type)(in_chan)
        #bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        #self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)
        # Succession of Conv1DBlock with exponentially increasing dilation.
        self.TCN = nn.ModuleList()
        for r in range(n_repeats):
            for x in range(n_blocks):
                padding = (kernel_size - 1) * 2 ** x // 2
                self.TCN.append(Conv1DBlock(in_chan, #TODO ask if also skip connections are used (probably not)
                                            kernel_size, padding=padding,
                                            dilation=2 ** x, norm_type=norm_type))
        mask_conv = nn.Conv1d(in_chan, n_src * in_chan, 1)
        self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)

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
        output = mixture_w
        #output = self.bottleneck(mixture_w)
        for i in range(len(self.TCN)):
            residual = self.TCN[i](output)
            output = output + residual
        emb = self.mask_net(output)
        emb = emb.view(batch, self.n_src, self.in_chan, n_frames)
        emb = emb / torch.sqrt(torch.sum(emb**2, 2, keepdim=True))
        return emb

    def get_config(self):
        config = {
            'in_chan': self.in_chan,
            'kernel_size': self.kernel_size,
            'n_blocks': self.n_blocks,
            'n_repeats': self.n_repeats,
            'n_src': self.n_src,
            'norm_type': self.norm_type,
        }
        return config


class SeparationStack(nn.Module):
    # basically this is plain conv-tasnet remove this in future releases

    def __init__(self, in_chan,  n_blocks=10, n_repeats=4,
                 kernel_size=3,
                 norm_type="gLN", mask_act="sigmoid"):

        super(SeparationStack, self).__init__()
        self.in_chan = in_chan
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.kernel_size = kernel_size
        self.norm_type = norm_type

        # layer_norm = norms.get(norm_type)(in_chan)
        # bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        # self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)
        # Succession of Conv1DBlock with exponentially increasing dilation.
        self.TCN = nn.ModuleList()
        for r in range(n_repeats):
            for x in range(n_blocks):
                padding = (kernel_size - 1) * 2 ** x // 2
                self.TCN.append(SepConv1DBlock(in_chan, in_chan,  # TODO ask if also skip connections are used (probably not)
                                            kernel_size, padding=padding,
                                            dilation=2 ** x, norm_type=norm_type))
        mask_conv = nn.Conv1d(in_chan, in_chan, 1)
        self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)

        # Get activation function.
        mask_nl_class = activations.get(mask_act)
        if has_arg(mask_nl_class, 'dim'):
            self.output_act = mask_nl_class(dim=1)
        else:
            self.output_act = mask_nl_class()

    def forward(self, mixture_w, spk_vectors):
        """
            Args:
                mixture_w (:class:`torch.Tensor`): Tensor of shape
                    [batch, n_filters, n_frames]

            Returns:
                :class:`torch.Tensor`:
                    estimated mask of shape [batch, n_src, n_filters, n_frames]
        """
        output = mixture_w
        # output = self.bottleneck(mixture_w)
        for i in range(len(self.TCN)):
            residual = self.TCN[i](output, spk_vectors)
            output = output + residual
        mask = self.mask_net(output)
        return self.output_act(mask)

    def get_config(self):
        config = {
            'in_chan': self.in_chan,
            'kernel_size': self.kernel_size,
            'n_blocks': self.n_blocks,
            'n_repeats': self.n_repeats,
            'norm_type': self.norm_type,
        }
        return config





