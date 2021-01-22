import torch
from asteroid_filterbanks.transforms import from_torch_complex, to_torch_complex
from ..masknn.recurrent import DCCRMaskNet
from .dcunet import BaseDCUNet


class DCCRNet(BaseDCUNet):
    """DCCRNet as proposed in [1].

    Args:
        architecture (str): The architecture to use, must be "DCCRN-CL".
        stft_kernel_size (int): STFT frame length to use
        stft_stride (int, optional): STFT hop length to use.
        sample_rate (float): Sampling rate of the model.
        masknet_kwargs (optional): Passed to :class:`DCCRMaskNet`

    References
        - [1] : "DCCRN: Deep Complex Convolution Recurrent Network for Phase-Aware Speech Enhancement",
          Yanxin Hu et al. https://arxiv.org/abs/2008.00264
    """

    masknet_class = DCCRMaskNet

    def __init__(
        self, *args, stft_n_filters=512, stft_kernel_size=400, stft_stride=100, **masknet_kwargs
    ):
        masknet_kwargs.setdefault("n_freqs", stft_n_filters // 2)
        super().__init__(
            *args,
            stft_n_filters=stft_n_filters,
            stft_kernel_size=stft_kernel_size,
            stft_stride=stft_stride,
            **masknet_kwargs,
        )

    def forward_encoder(self, wav):
        tf_rep = self.encoder(wav)
        # Remove Nyquist frequency bin
        return to_torch_complex(tf_rep)[..., :-1, :]

    def apply_masks(self, tf_rep, est_masks):
        masked_tf_rep = est_masks * tf_rep.unsqueeze(1)
        # Pad Nyquist frequency bin
        return from_torch_complex(torch.nn.functional.pad(masked_tf_rep, [0, 0, 0, 1]))
