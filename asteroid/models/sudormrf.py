import torch
from torch import nn
import math

from asteroid_filterbanks import make_enc_dec
from ..masknn import SuDORMRF, SuDORMRFImproved
from .base_models import BaseEncoderMaskerDecoder
from ..utils.torch_utils import script_if_tracing


class SuDORMRFNet(BaseEncoderMaskerDecoder):
    """SuDORMRF separation model, as described in [1].

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
        stride (int, optional): Stride of the convolution.
            If None (default), set to ``kernel_size // 2``.
        sample_rate (float): Sampling rate of the model.
        **fb_kwargs (dict): Additional kwards to pass to the filterbank
            creation.

    References
        - [1] : "Sudo rm -rf: Efficient Networks for Universal Audio Source Separation",
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
        stride=None,
        sample_rate=8000,
        **fb_kwargs,
    ):
        # Need the encoder to determine the number of input channels
        stride = kernel_size // 2 if not stride else stride
        enc, dec = make_enc_dec(
            fb_name,
            kernel_size=kernel_size,
            n_filters=n_filters,
            stride=kernel_size // 2,
            sample_rate=sample_rate,
            padding=kernel_size // 2,
            output_padding=(kernel_size // 2) - 1,
            **fb_kwargs,
        )
        n_feats = enc.n_feats_out
        enc = _Padder(enc, upsampling_depth=upsampling_depth, kernel_size=kernel_size)

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


class SuDORMRFImprovedNet(BaseEncoderMaskerDecoder):
    """Improved SuDORMRF separation model, as described in [1].

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
        stride (int, optional): Stride of the convolution.
            If None (default), set to ``kernel_size // 2``.
        **fb_kwargs (dict): Additional kwards to pass to the filterbank
            creation.

    References
        - [1] : "Sudo rm -rf: Efficient Networks for Universal Audio Source Separation",
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
        stride=None,
        sample_rate=8000,
        **fb_kwargs,
    ):
        stride = kernel_size // 2 if not stride else stride
        # Need the encoder to determine the number of input channels
        enc, dec = make_enc_dec(
            fb_name,
            kernel_size=kernel_size,
            n_filters=n_filters,
            stride=stride,
            sample_rate=sample_rate,
            padding=kernel_size // 2,
            output_padding=(kernel_size // 2) - 1,
            **fb_kwargs,
        )
        n_feats = enc.n_feats_out
        enc = _Padder(enc, upsampling_depth=upsampling_depth, kernel_size=kernel_size)

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


class _Padder(nn.Module):
    def __init__(self, encoder, upsampling_depth=4, kernel_size=21):
        super().__init__()
        self.encoder = encoder
        self.upsampling_depth = upsampling_depth
        self.kernel_size = kernel_size

        # Appropriate padding is needed for arbitrary lengths
        self.lcm = abs(self.kernel_size // 2 * 2 ** self.upsampling_depth) // math.gcd(
            self.kernel_size // 2, 2 ** self.upsampling_depth
        )

        # For serialize
        self.filterbank = self.encoder.filterbank
        self.sample_rate = getattr(self.encoder.filterbank, "sample_rate", None)

    def forward(self, x):
        x = pad(x, self.lcm)
        return self.encoder(x)


@script_if_tracing
def pad(x, lcm: int):
    values_to_pad = int(x.shape[-1]) % lcm
    if values_to_pad:
        appropriate_shape = x.shape
        padding = torch.zeros(
            list(appropriate_shape[:-1]) + [lcm - values_to_pad],
            dtype=x.dtype,
        )
        padded_x = torch.cat([x, padding], dim=-1)
        return padded_x
    return x
