import torch
from torch import nn
from copy import deepcopy

from asteroid_filterbanks import make_enc_dec
from ..masknn import LSTMMasker
from .base_models import BaseEncoderMaskerDecoder


class LSTMTasNet(BaseEncoderMaskerDecoder):
    """TasNet separation model, as described in [1].

    Args:
        n_src (int): Number of masks to estimate.
        out_chan  (int or None): Number of bins in the estimated masks.
            Defaults to `in_chan`.
        hid_size (int): Number of neurons in the RNNs cell state.
            Defaults to 128.
        mask_act (str, optional): Which non-linear function to generate mask.
        bidirectional (bool, optional): True for bidirectional Inter-Chunk RNN
            (Intra-Chunk is always bidirectional).
        rnn_type (str, optional): Type of RNN used. Choose between ``'RNN'``,
            ``'LSTM'`` and ``'GRU'``.
        n_layers (int, optional): Number of layers in each RNN.
        dropout (float, optional): Dropout ratio, must be in [0,1].
        in_chan (int, optional): Number of input channels, should be equal to
            n_filters.
        fb_name (str, className): Filterbank family from which to make encoder
            and decoder. To choose among [``'free'``, ``'analytic_free'``,
            ``'param_sinc'``, ``'stft'``].
        n_filters (int): Number of filters / Input dimension of the masker net.
        kernel_size (int): Length of the filters.
        stride (int, optional): Stride of the convolution.
            If None (default), set to ``kernel_size // 2``.
        sample_rate (float): Sampling rate of the model.
        **fb_kwargs (dict): Additional kwards to pass to the filterbank
            creation.

    References
        - [1]: Yi Luo et al. "Real-time Single-channel Dereverberation and Separation
          with Time-domain Audio Separation Network", Interspeech 2018
    """

    def __init__(
        self,
        n_src,
        out_chan=None,
        rnn_type="lstm",
        n_layers=4,
        hid_size=512,
        dropout=0.3,
        mask_act="sigmoid",
        bidirectional=True,
        in_chan=None,
        fb_name="free",
        n_filters=64,
        kernel_size=16,
        stride=8,
        encoder_activation=None,
        sample_rate=8000,
        **fb_kwargs,
    ):
        encoder, decoder = make_enc_dec(
            fb_name,
            kernel_size=kernel_size,
            n_filters=n_filters,
            stride=stride,
            sample_rate=sample_rate,
            **fb_kwargs,
        )
        n_feats = encoder.n_feats_out
        if in_chan is not None:
            assert in_chan == n_feats, (
                "Number of filterbank output channels"
                " and number of input channels should "
                "be the same. Received "
                f"{n_feats} and {in_chan}"
            )

        # Real gated encoder
        encoder = _GatedEncoder(encoder)

        # Masker
        masker = LSTMMasker(
            n_feats,
            n_src,
            out_chan=out_chan,
            hid_size=hid_size,
            mask_act=mask_act,
            bidirectional=bidirectional,
            rnn_type=rnn_type,
            n_layers=n_layers,
            dropout=dropout,
        )
        super().__init__(encoder, masker, decoder, encoder_activation=encoder_activation)


class _GatedEncoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        # For config
        self.filterbank = encoder.filterbank
        self.sample_rate = getattr(encoder.filterbank, "sample_rate", None)
        # Gated encoder.
        self.encoder_relu = encoder
        self.encoder_sig = deepcopy(encoder)

    def forward(self, x):
        relu_out = torch.relu(self.encoder_relu(x))
        sig_out = torch.sigmoid(self.encoder_sig(x))
        return sig_out * relu_out
