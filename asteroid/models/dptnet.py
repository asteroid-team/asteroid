from ..filterbanks import make_enc_dec
from ..masknn import DPTransformer
from .base_models import BaseTasNet


class DPTNet(BaseTasNet):
    """ DPTNet separation model, as described in [1].

    Args:
        n_src (int): Number of masks to estimate.
        out_chan  (int or None): Number of bins in the estimated masks.
            Defaults to `in_chan`.
        bn_chan (int): Number of channels after the bottleneck.
            Defaults to 128.
        hid_size (int): Number of neurons in the RNNs cell state.
            Defaults to 128.
        chunk_size (int): window size of overlap and add processing.
            Defaults to 100.
        hop_size (int or None): hop size (stride) of overlap and add processing.
            Default to `chunk_size // 2` (50% overlap).
        n_repeats (int): Number of repeats. Defaults to 6.
        norm_type (str, optional): Type of normalization to use. To choose from

            - ``'gLN'``: global Layernorm
            - ``'cLN'``: channelwise Layernorm
        mask_act (str, optional): Which non-linear function to generate mask.
        bidirectional (bool, optional): True for bidirectional Inter-Chunk RNN
            (Intra-Chunk is always bidirectional).
        rnn_type (str, optional): Type of RNN used. Choose between ``'RNN'``,
            ``'LSTM'`` and ``'GRU'``.
        num_layers (int, optional): Number of layers in each RNN.
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
        **fb_kwargs (dict): Additional kwards to pass to the filterbank
            creation.

    References:
        [1]: Jingjing Chen et al. "Dual-Path Transformer Network: Direct
            Context-Aware Modeling for End-to-End Monaural Speech Separation"
            Interspeech 2020.
    """

    def __init__(
        self,
        n_src,
        ff_hid=256,
        chunk_size=100,
        hop_size=None,
        n_repeats=6,
        norm_type="gLN",
        ff_activation="relu",
        encoder_activation="relu",
        mask_act="relu",
        bidirectional=True,
        dropout=0,
        in_chan=None,
        fb_name="free",
        kernel_size=16,
        n_filters=64,
        stride=8,
        **fb_kwargs,
    ):
        encoder, decoder = make_enc_dec(
            fb_name, kernel_size=kernel_size, n_filters=n_filters, stride=stride, **fb_kwargs
        )
        n_feats = encoder.n_feats_out
        if in_chan is not None:
            assert in_chan == n_feats, (
                "Number of filterbank output channels"
                " and number of input channels should "
                "be the same. Received "
                f"{n_feats} and {in_chan}"
            )
        # Update in_chan
        masker = DPTransformer(
            n_feats,
            n_src,
            ff_hid=ff_hid,
            ff_activation=ff_activation,
            chunk_size=chunk_size,
            hop_size=hop_size,
            n_repeats=n_repeats,
            norm_type=norm_type,
            mask_act=mask_act,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        super().__init__(encoder, masker, decoder, encoder_activation=encoder_activation)
