import torch
from .. import complex_nn
from ..filterbanks.transforms import from_torchaudio
from ..masknn.recurrent import DCCRMaskNet
from .dcunet import BaseDCUNet


class DCCRNet(BaseDCUNet):  # CHECK-JIT
    """DCCRNet as proposed in [1].

    Args:
        architecture (str): The architecture to use, must be "DCCRN-CL".
        stft_kernel_size (int): STFT frame length to use
        stft_stride (int, optional): STFT hop length to use.
        sample_rate (float): Sampling rate of the model.

    References
        - [1] : "DCCRN: Deep Complex Convolution Recurrent Network for Phase-Aware Speech Enhancement",
        Yanxin Hu et al. https://arxiv.org/abs/2008.00264
    """

    masknet_class = DCCRMaskNet

    def __init__(
        self, *args, stft_kernel_size=512, sample_rate=16000.0, masknet_kwargs=None, **kwargs
    ):
        super().__init__(
            *args,
            stft_kernel_size=stft_kernel_size,
            sample_rate=sample_rate,
            masknet_kwargs={"n_freqs": stft_kernel_size // 2, **(masknet_kwargs or {})},
            **kwargs,
        )

    def forward_encoder(self, wav):
        tf_rep = self.encoder(wav)
        # Remove Nyquist frequency bin
        return complex_nn.as_torch_complex(tf_rep)[..., :-1, :]

    def apply_masks(self, tf_rep, est_masks):
        masked_tf_rep = est_masks * tf_rep.unsqueeze(1)
        # Pad Nyquist frequency bin
        return from_torchaudio(
            torch.view_as_real(torch.nn.functional.pad(masked_tf_rep, (0, 0, 0, 1)))
        )
