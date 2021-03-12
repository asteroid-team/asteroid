import json
import os
import torch
from torch import nn
from torch.nn.functional import fold, unfold


from asteroid_filterbanks import make_enc_dec
from asteroid.engine.optimizers import make_optimizer
from asteroid.masknn import activations, norms
from asteroid.masknn.recurrent import DPRNNBlock
from asteroid.models.base_models import _shape_reconstructed, _unsqueeze_to_3d
from asteroid.utils.generic_utils import has_arg
from asteroid.utils.torch_utils import pad_x_to_y, script_if_tracing, jitable_shape


def make_model_and_optimizer(conf):
    """Function to define the model and optimizer for a config dictionary.
    Args:
        conf: Dictionary containing the output of hierachical argparse.
    Returns:
        model, optimizer.
    The main goal of this function is to make reloading for resuming
    and evaluation very simple.
    """
    model = MultiDecoderDPRNN(**conf["masknet"], **conf["filterbank"])
    optimizer = make_optimizer(model.parameters(), **conf["optim"])
    return model, optimizer


class DPRNN_MultiStage(nn.Module):
    """Implementation of the Dual-Path-RNN model,
        with multi-stage output, without Conv2D projection

    Args:
        in_chan: The number of expected features in the input x
        out_channels: The number of features in the hidden state h
        rnn_type: RNN, LSTM, GRU
        norm: gln = "Global Norm", cln = "Cumulative Norm", ln = "Layer Norm"
        dropout: If non-zero, introduces a Dropout layer on the outputs
                    of each LSTM layer except the last layer,
                    with dropout probability equal to dropout. Default: 0
        bidirectional: If True, becomes a bidirectional LSTM. Default: False
        num_layers: number of Dual-Path-Block
        K: the length of chunk
        num_spks: the number of speakers
    """

    def __init__(
        self,
        in_chan,
        bn_chan,
        hid_size,
        chunk_size,
        hop_size,
        n_repeats,
        norm_type,
        bidirectional,
        rnn_type,
        use_mulcat,
        num_layers,
        dropout,
    ):
        super(DPRNN_MultiStage, self).__init__()
        self.in_chan = in_chan
        self.bn_chan = bn_chan
        self.hid_size = hid_size
        self.chunk_size = chunk_size
        self.hop_size = hop_size
        self.n_repeats = n_repeats
        self.norm_type = norm_type
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_mulcat = use_mulcat
        self.num_layers = num_layers

        layer_norm = norms.get(norm_type)(in_chan)
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)

        # Succession of DPRNNBlocks.
        self.net = nn.ModuleList([])
        for i in range(self.n_repeats):
            self.net.append(
                DPRNNBlock(
                    bn_chan,
                    hid_size,
                    norm_type=norm_type,
                    bidirectional=bidirectional,
                    rnn_type=rnn_type,
                    use_mulcat=use_mulcat,
                    num_layers=num_layers,
                    dropout=dropout,
                )
            )

    def forward(self, mixture_w):
        r"""Forward.

        Args:
            mixture_w (:class:`torch.Tensor`): Tensor of shape $(batch, nfilters, nframes)$

        Returns:
            list of (:class:`torch.Tensor`): Tensor of shape $(batch, bn_chan, chunk_size, n_chunks)
        """
        batch, n_filters, n_frames = mixture_w.size()
        output = self.bottleneck(mixture_w)  # [batch, bn_chan, n_frames]
        output = unfold(
            output.unsqueeze(-1),
            kernel_size=(self.chunk_size, 1),
            padding=(self.chunk_size, 0),
            stride=(self.hop_size, 1),
        )
        n_chunks = output.shape[-1]
        output = output.reshape(batch, self.bn_chan, self.chunk_size, n_chunks)
        # Apply stacked DPRNN Blocks sequentially
        output_list = []
        for i in range(self.n_repeats):
            output = self.net[i](output)
            output_list.append(output)
        return output_list


class SingleDecoder(nn.Module):
    def __init__(
        self, kernel_size, stride, in_chan, n_src, bn_chan, chunk_size, hop_size, mask_act
    ):
        super(SingleDecoder, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_chan = in_chan
        self.bn_chan = bn_chan
        self.chunk_size = chunk_size
        self.hop_size = hop_size
        self.n_src = n_src
        self.mask_act = mask_act

        # Masking in 3D space
        net_out_conv = nn.Conv2d(bn_chan, n_src * bn_chan, 1)
        self.first_out = nn.Sequential(nn.PReLU(), net_out_conv)
        # Gating and masking in 2D space (after fold)
        self.net_out = nn.Sequential(nn.Conv1d(bn_chan, bn_chan, 1), nn.Tanh())
        self.net_gate = nn.Sequential(nn.Conv1d(bn_chan, bn_chan, 1), nn.Sigmoid())
        self.mask_net = nn.Conv1d(bn_chan, in_chan, 1, bias=False)

        # Get activation function.
        mask_nl_class = activations.get(mask_act)
        # For softmax, feed the source dimension.
        if has_arg(mask_nl_class, "dim"):
            self.output_act = mask_nl_class(dim=1)
        else:
            self.output_act = mask_nl_class()

        _, self.trans_conv = make_enc_dec(
            "free", kernel_size=kernel_size, stride=stride, n_filters=in_chan
        )

    def forward(self, output, mixture_w):
        """
        Takes a single example mask and encoding, outputs waveform
        Args:
            output: LSTM output, Tensor of shape $(num_stages, bn_chan, chunk_size, n_chunks)$
            mixture_w: Encoder output, Tensor of shape $(num_stages, in_chan, nframes)
        outputs:
            Signal, Tensor of shape $(num_stages, n_src, T)
        """
        batch, bn_chan, chunk_size, n_chunks = output.size()
        _, in_chan, n_frames = mixture_w.size()
        assert self.bn_chan == bn_chan
        assert self.in_chan == in_chan
        assert self.chunk_size == chunk_size
        output = self.first_out(output)
        output = output.reshape(batch * self.n_src, self.bn_chan, self.chunk_size, n_chunks)
        # Overlap and add:
        # [batch, out_chan, chunk_size, n_chunks] -> [batch, out_chan, n_frames]
        to_unfold = self.bn_chan * self.chunk_size
        output = fold(
            output.reshape(batch * self.n_src, to_unfold, n_chunks),
            (n_frames, 1),
            kernel_size=(self.chunk_size, 1),
            padding=(self.chunk_size, 0),
            stride=(self.hop_size, 1),
        )
        # Apply gating
        output = output.reshape(batch * self.n_src, self.bn_chan, -1)
        output = self.net_out(output) * self.net_gate(output)
        # Compute mask
        score = self.mask_net(output)
        est_mask = self.output_act(score)
        est_mask = est_mask.reshape(batch, self.n_src, self.in_chan, n_frames)
        mixture_w = mixture_w.unsqueeze(1)
        source_w = est_mask * mixture_w
        source_w = source_w.reshape(batch * self.n_src, self.in_chan, n_frames)
        est_wavs = self.trans_conv(source_w)
        est_wavs = est_wavs.reshape(batch, self.n_src, -1)
        return est_wavs


class Decoder_Select(nn.Module):
    """Selects which decoder to use, as well as whether to use multiloss
    Args:
        n_srcs: list of [B], number of sources for each decoder
    """

    def __init__(
        self, kernel_size, stride, in_chan, n_srcs, bn_chan, chunk_size, hop_size, mask_act
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_chan = in_chan
        self.n_srcs = n_srcs
        self.bn_chan = bn_chan
        self.chunk_size = chunk_size
        self.hop_size = hop_size
        self.mask_act = mask_act

        self.n_src2idx = {n_src: i for i, n_src in enumerate(n_srcs)}
        self.decoders = torch.nn.ModuleList()
        for n_src in n_srcs:
            self.decoders.append(
                SingleDecoder(
                    kernel_size=kernel_size,
                    stride=stride,
                    in_chan=in_chan,
                    n_src=n_src,
                    bn_chan=bn_chan,
                    chunk_size=chunk_size,
                    hop_size=hop_size,
                    mask_act=mask_act,
                )
            )
        self.selector = nn.Sequential(
            nn.Conv2d(bn_chan, in_chan, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.ReLU(),
            nn.Conv2d(in_chan, len(n_srcs), 1),
        )

    def forward(self, output_list, mixture_w, ground_truth):
        """Forward
        Args:
            output_list: list of $(batch, bn_chan, chunk_size, n_chunks)$
            mixture_w: Tensor of $(batch, in_chan, n_frames)$
            ground_truth: None of list of [B] ints, or Long Tensor of $(B)
        Output:
            output_wavs: Tensor of $(batch, num_stages, maxspks, T)$
            selector_output: $(batch, num_stages, num_decoders)$
        """
        batch, bn_chan, chunk_size, n_chunks = output_list[0].size()
        _, in_chan, n_frames = mixture_w.size()
        assert self.chunk_size == chunk_size
        if not self.training:
            output_list = output_list[-1:]
        num_stages = len(output_list)
        # [batch, num_stages, bn_chan, chunk_size, n_chunks]
        output = torch.stack(output_list, 1).reshape(
            batch * num_stages, bn_chan, chunk_size, n_chunks
        )
        selector_output = self.selector(output).reshape(batch, num_stages, -1)
        output = output.reshape(batch, num_stages, bn_chan, chunk_size, n_chunks)
        # [batch, num_stages, in_chan, n_frames]
        mixture_w = mixture_w.unsqueeze(1).repeat(1, num_stages, 1, 1)
        if ground_truth is not None:  # oracle
            decoder_selected = torch.LongTensor([self.n_src2idx[truth] for truth in ground_truth])
        else:
            assert num_stages == 1  # can't use select with multistage
            decoder_selected = selector_output.reshape(batch, -1).argmax(1)
        T = self.kernel_size + self.stride * (n_frames - 1)
        output_wavs = torch.zeros(batch, num_stages, max(self.n_srcs), T).to(output.device)
        for i in range(batch):
            output_wavs[i, :, : self.n_srcs[decoder_selected[i]], :] = self.decoders[
                decoder_selected[i]
            ](output[i], mixture_w[i])
        return output_wavs, selector_output


class MultiDecoderDPRNN(nn.Module):
    def __init__(
        self,
        n_srcs,
        bn_chan=128,
        hid_size=128,
        chunk_size=100,
        hop_size=None,
        n_repeats=6,
        norm_type="gLN",
        mask_act="sigmoid",
        bidirectional=True,
        rnn_type="LSTM",
        num_layers=1,
        dropout=0,
        kernel_size=16,
        n_filters=64,
        stride=8,
        encoder_activation=None,
        use_mulcat=False,
    ):
        super().__init__()
        self.encoder_activation = encoder_activation
        self.enc_activation = activations.get(encoder_activation or "linear")()
        hop_size = hop_size if hop_size is not None else chunk_size // 2
        self.encoder, _ = make_enc_dec(
            "free",
            kernel_size=kernel_size,
            n_filters=n_filters,
            stride=stride,
        )
        # Update in_chan
        self.masker = DPRNN_MultiStage(
            in_chan=n_filters,
            bn_chan=bn_chan,
            hid_size=hid_size,
            chunk_size=chunk_size,
            hop_size=hop_size,
            n_repeats=n_repeats,
            norm_type=norm_type,
            bidirectional=bidirectional,
            rnn_type=rnn_type,
            use_mulcat=use_mulcat,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.decoder_select = Decoder_Select(
            kernel_size=kernel_size,
            stride=stride,
            in_chan=n_filters,
            n_srcs=n_srcs,
            bn_chan=bn_chan,
            chunk_size=chunk_size,
            hop_size=hop_size,
            mask_act=mask_act,
        )

    def forward(self, wav, ground_truth=None):
        shape = jitable_shape(wav)
        # [batch, 1, T]
        wav = _unsqueeze_to_3d(wav)
        tf_rep = self.enc_activation(self.encoder(wav))
        est_masks_list = self.masker(tf_rep)
        decoded, selector_output = self.decoder_select(
            est_masks_list, tf_rep, ground_truth=ground_truth
        )
        reconstructed = pad_x_to_y(decoded, wav)
        return _shape_reconstructed(reconstructed, shape), _shape_reconstructed(
            selector_output, shape
        )


# Training notes:
# Weight different stages in accordance with facebook code
if __name__ == "__main__":
    network = MultiDecoderDPRNN([2, 3], bn_chan=32, hid_size=32, n_filters=16)
    input = torch.rand(2, 3200)
    wavs, selector_output = network(input, [3, 2])
    print(wavs.shape)
    assert (wavs[1, :, 2] == 0).all()
    network.eval()
    wavs, selector_output = network(input)
    print(wavs.shape)
