import numpy as np
import torch

from .. import complex_nn


class BaseUNet(torch.nn.Module):
    """Base class for u-nets with skip connections between encoders and decoders.

    (For u-nets without skip connections, simply use a `nn.Sequential`.)

    Args:
        encoders (List[torch.nn.Module]): List of encoders
        decoders (List[torch.nn.Module]): List of decoders
        intermediate_layer (Optional[torch.nn.Module], optional):
            Layer between last encoder and first decoder.
        output_layer (Optional[torch.nn.Module], optional):
            Layer after last decoder.
    """

    def __init__(
        self,
        encoders,
        decoders,
        *,
        intermediate_layer=None,
        output_layer=None,
    ):
        super().__init__()

        self.encoders = torch.nn.ModuleList(encoders)
        self.decoders = torch.nn.ModuleList(decoders)
        self.intermediate_layer = intermediate_layer or torch.nn.Identity()
        self.output_layer = output_layer or torch.nn.Identity()

    def forward(self, x):
        enc_outs = []
        for idx, enc in enumerate(self.encoders):
            x = enc(x)
            enc_outs.append(x)
        x = self.intermediate_layer(x)
        for idx, (enc_out, dec) in enumerate(zip(reversed(enc_outs[:-1]), self.decoders)):
            x = dec(x)
            x = torch.cat([x, enc_out], dim=1)
        return self.output_layer(x)


class BaseDCUMaskNet(BaseUNet):
    """Base class for DCU-style mask nets. Used for DCUMaskNet and DCCRMaskNet.

    The preferred way to instantiate this class is to use the ``default_architecture()``
    classmethod.

    Args:
        encoders (list of length `N` of tuples of (in_chan, out_chan, kernel_size, stride, padding)):
            Arguments of encoders of the u-net
        decoders (list of length `N` of tuples of (in_chan, out_chan, kernel_size, stride, padding))
            Arguments of decoders of the u-net
        mask_bound (Optional[str], optional): Type of mask bound to use, as defined in [1].
            Valid values are "tanh" ("BDT mask"), "sigmoid" ("BDSS mask"), None (unbounded mask).

    Input shape is expected to be [batch, n_freqs, time], with `n_freqs - 1` divisible
    by `f_0 * f_1 * ... * f_N` where `f_k` are the frequency strides of the encoders.

    References
        - [1] : "Phase-aware Speech Enhancement with Deep Complex U-Net",
        Hyeong-Seok Choi et al. https://arxiv.org/abs/1903.03107
    """

    _architectures = NotImplemented

    @classmethod
    def default_architecture(cls, architecture: str, **kwargs):
        """Create a masknet instance from a predefined, named architecture.

        Args:
            architecture (str): Name of predefined architecture. Valid values
                are dependent on the concrete subclass of ``BaseDCUMaskNet``.
            kwargs (optional): Passed to ``__init__`.
        """
        encoders, decoders = cls._architectures[architecture]
        return cls(encoders, decoders, **kwargs)

    def __init__(self, encoders, decoders, mask_bound="tanh", **kwargs):
        self.encoder_args = encoders
        self.decoder_args = decoders
        self.mask_bound = mask_bound

        # Avoid circual import
        from .convolutional import DCUNetComplexDecoderBlock, DCUNetComplexEncoderBlock

        super().__init__(
            encoders=[DCUNetComplexEncoderBlock(*args) for args in encoders],
            decoders=[DCUNetComplexDecoderBlock(*args) for args in decoders],
            output_layer=torch.nn.Sequential(
                complex_nn.ComplexConvTranspose2d(*decoders[-1]),
                complex_nn.BoundComplexMask(mask_bound),
            ),
            **kwargs,
        )

    @property
    def encoders_stride_product(self):
        return np.prod([enc_stride for _, _, _, enc_stride, _ in self.encoder_args], axis=0)

    @property
    def decoders_stride_product(self):
        return np.prod([enc_stride for _, _, _, enc_stride, _ in self.decoder_args], axis=0)

    def forward(self, x):
        # TODO: We can probably lift the shape requirements once Keras-style "same"
        # padding for convolutions has landed: https://github.com/pytorch/pytorch/pull/42190
        freq_prod, time_prod = self.encoders_stride_product
        if (x.shape[1] - 1) % freq_prod or (x.shape[2] - 1) % time_prod:
            raise TypeError(
                f"Input shape must be [batch, freq + 1, time + 1] with freq divisible by "
                f"{freq_prod} and time divisible by {time_prod}, got {x.shape} instead"
            )
        return super().forward(x.unsqueeze(1))
