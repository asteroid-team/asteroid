from ..filterbanks import make_enc_dec
from ..masknn import SuDORMRF, SuDORMRFImproved
from .base_models import BaseTasNet


class SuDORMRFNet(BaseTasNet):
    """ SuDORMRF separation model, as described in [1].

    Args:
        n_src (int): Number of sources in the input mixtures.
        bn_chan (int, optional): Number of bins in the bottleneck layer and the UNet blocks.
        num_blocks (int): Number of of UBlocks.
        upsampling_depth (int): Depth of upsampling.
        mask_act (str): Name of output activation.
        in_chan (int, optional): Number of input channels, should be equal to
            n_filters.
        fb_name (str, className): Filterbank family from which to make encoder
            and decoder. To choose among [``'free'``, ``'analytic_free'``,
            ``'param_sinc'``, ``'stft'``].
        n_filters (int): Number of filters / Input dimension of the masker net.
        kernel_size (int): Length of the filters.
        **fb_kwargs (dict): Additional kwards to pass to the filterbank
            creation.

    References:
        [1] : "Sudo rm -rf: Efficient Networks for Universal Audio Source Separation",
            Tzinis et al. MLSP 2020.
    """

    def __init__(
        self,
        n_src,
        bn_chan=128,
        num_blocks=16,
        upsampling_depth=4,
        mask_act="softmax",
        in_chan=None,
        fb_name="free",
        kernel_size=21,
        n_filters=512,
        **fb_kwargs,
    ):
        # Need the encoder to determine the number of input channels
        enc, dec = make_enc_dec(
            fb_name,
            kernel_size=kernel_size,
            n_filters=n_filters,
            stride=kernel_size // 2,
            padding=kernel_size // 2,
            output_padding=(kernel_size // 2) - 1,
            **fb_kwargs,
        )
        n_feats = enc.n_feats_out
        if in_chan is not None:
            assert in_chan == n_feats, (
                "Number of filterbank output channels"
                " and number of input channels should "
                "be the same. Received "
                f"{n_feats} and {in_chan}"
            )
        masker = SuDORMRF(
            n_feats,
            n_src,
            bn_chan=bn_chan,
            num_blocks=num_blocks,
            upsampling_depth=upsampling_depth,
            mask_act=mask_act,
        )
        super().__init__(enc, masker, dec, encoder_activation="relu")


class SuDORMRFImprovedNet(BaseTasNet):
    """ Improved SuDORMRF separation model, as described in [1].

    Args:
        n_src (int): Number of sources in the input mixtures.
        bn_chan (int, optional): Number of bins in the bottleneck layer and the UNet blocks.
        num_blocks (int): Number of of UBlocks.
        upsampling_depth (int): Depth of upsampling.
        mask_act (str): Name of output activation.
        in_chan (int, optional): Number of input channels, should be equal to
            n_filters.
        fb_name (str, className): Filterbank family from which to make encoder
            and decoder. To choose among [``'free'``, ``'analytic_free'``,
            ``'param_sinc'``, ``'stft'``].
        n_filters (int): Number of filters / Input dimension of the masker net.
        kernel_size (int): Length of the filters.
        **fb_kwargs (dict): Additional kwards to pass to the filterbank
            creation.

    References:
        [1] : "Sudo rm -rf: Efficient Networks for Universal Audio Source Separation",
            Tzinis et al. MLSP 2020.
    """

    def __init__(
        self,
        n_src,
        bn_chan=128,
        num_blocks=16,
        upsampling_depth=4,
        mask_act="relu",
        in_chan=None,
        fb_name="free",
        kernel_size=21,
        n_filters=512,
        **fb_kwargs,
    ):

        # Need the encoder to determine the number of input channels
        enc, dec = make_enc_dec(
            fb_name,
            kernel_size=kernel_size,
            n_filters=n_filters,
            stride=kernel_size // 2,
            padding=kernel_size // 2,
            output_padding=(kernel_size // 2) - 1,
            **fb_kwargs,
        )
        n_feats = enc.n_feats_out
        if in_chan is not None:
            assert in_chan == n_feats, (
                "Number of filterbank output channels"
                " and number of input channels should "
                "be the same. Received "
                f"{n_feats} and {in_chan}"
            )
        masker = SuDORMRFImproved(
            n_feats,
            n_src,
            bn_chan=bn_chan,
            num_blocks=num_blocks,
            upsampling_depth=upsampling_depth,
            mask_act=mask_act,
        )
        super().__init__(enc, masker, dec, encoder_activation=None)
