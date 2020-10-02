import torch

from .. import complex_nn
from ..filterbanks import make_enc_dec
from ..filterbanks.transforms import from_torchaudio
from ..masknn.convolutional import DCUMaskNet
from .base_models import BaseEncoderMaskerDecoder


class BaseDCUNet(BaseEncoderMaskerDecoder):
    """Base class for ``DCUNet`` and ``DCCRNet`` classes.

    Args:
        stft_kernel_size (int): STFT frame length to use
        stft_stride (int, optional): STFT hop length to use.
    """

    masknet_class = DCUMaskNet

    def __init__(self, architecture, stft_kernel_size=512, stft_stride=None, masknet_kwargs=None):
        self.architecture = architecture
        self.stft_kernel_size = stft_kernel_size
        self.stft_stride = stft_stride
        self.masknet_kwargs = masknet_kwargs

        encoder, decoder = make_enc_dec(
            "stft", kernel_size=stft_kernel_size, n_filters=stft_kernel_size, stride=stft_stride
        )
        masker = self.masknet_class.default_architecture(architecture, **(masknet_kwargs or {}))
        super().__init__(encoder, masker, decoder)

    def postprocess_encoded(self, tf_rep):
        return complex_nn.as_torch_complex(tf_rep)

    def postprocess_masked(self, masked_tf_rep):
        return from_torchaudio(torch.view_as_real(masked_tf_rep))

    def get_model_args(self):
        """Arguments needed to re-instantiate the model."""
        model_args = {
            "architecture": self.architecture,
            "stft_kernel_size": self.stft_kernel_size,
            "stft_stride": self.stft_stride,
            "masknet_kwargs": self.masknet_kwargs,
        }
        return model_args


class DCUNet(BaseDCUNet):
    """DCUNet as proposed in [1].

    Args:
        architecture (str): The architecture to use, any of
            "DCUNet-10", "DCUNet-16", "DCUNet-20", "Large-DCUNet-20".
        stft_kernel_size (int): STFT frame length to use
        stft_stride (int, optional): STFT hop length to use.

    References:
        [1] : "Phase-aware Speech Enhancement with Deep Complex U-Net",
        Hyeong-Seok Choi et al.
        https://arxiv.org/abs/1903.03107
    """

    masknet_class = DCUMaskNet
