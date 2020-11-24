import numpy as np
import torch

from .. import complex_nn


def _none_sequential(*args):
    return torch.nn.Sequential(*[x for x in args if x is not None])


class BaseUNet(torch.nn.Module):
    """Base class for u-nets with skip connections between encoders and decoders.

    (For u-nets without skip connections, simply use a `nn.Sequential`.)

    Args:
        encoders (List[torch.nn.Module] of length `N`): List of encoders
        decoders (List[torch.nn.Module] of length `N - 1`): List of decoders
        output_layer (Optional[torch.nn.Module], optional):
            Layer after last decoder.
    """

    def __init__(
        self,
        encoders,
        decoders,
        *,
        output_layer=None,
    ):
        assert len(encoders) == len(decoders) + 1

        super().__init__()

        self.encoders = torch.nn.ModuleList(encoders)
        self.decoders = torch.nn.ModuleList(decoders)
        self.output_layer = output_layer or torch.nn.Identity()

    def forward(self, x):
        enc_outs = []
        for idx, enc in enumerate(self.encoders):
            x = enc(x)
            enc_outs.append(x)
        for idx, (enc_out, dec) in enumerate(zip(reversed(enc_outs[:-1]), self.decoders)):
            x = dec(x)
            x = torch.cat([x, enc_out], dim=1)
        return self.output_layer(x)


class BaseDCUMaskNet(BaseUNet):
    """Base class for DCU-style mask nets. Used for DCUMaskNet and DCCRMaskNet.

    The preferred way to instantiate this class is to use the ``default_architecture()``
    classmethod.

    Args:
        encoders (List[torch.nn.Module]): List of encoders
        decoders (List[torch.nn.Module]): List of decoders
        output_layer (Optional[torch.nn.Module], optional):
            Layer after last decoder, before mask application.
        mask_bound (Optional[str], optional): Type of mask bound to use, as defined in [1].
            Valid values are "tanh" ("BDT mask"), "sigmoid" ("BDSS mask"), None (unbounded mask).

    References
        - [1] : "Phase-aware Speech Enhancement with Deep Complex U-Net",
        Hyeong-Seok Choi et al. https://arxiv.org/abs/1903.03107
    """

    _architectures = NotImplemented

    @classmethod
    def default_architecture(cls, architecture: str, n_src=1, **kwargs):
        """Create a masknet instance from a predefined, named architecture.

        Args:
            architecture (str): Name of predefined architecture. Valid values
                are dependent on the concrete subclass of ``BaseDCUMaskNet``.
            n_src (int, optional): Number of sources
            kwargs (optional): Passed to ``__init__``.
        """
        encoders, decoders = cls._architectures[architecture]
        # Fix n_src in last decoder
        in_chan, _ignored_out_chan, *rest = decoders[-1]
        decoders = (*decoders[:-1], (in_chan, n_src, *rest))
        return cls(encoders, decoders, **kwargs)

    def __init__(self, encoders, decoders, output_layer=None, mask_bound="tanh", **kwargs):
        self.mask_bound = mask_bound
        super().__init__(
            encoders=encoders,
            decoders=decoders,
            output_layer=_none_sequential(
                output_layer,
                complex_nn.BoundComplexMask(mask_bound),
            ),
            **kwargs,
        )

    def forward(self, x):
        fixed_x = self.fix_input_dims(x)
        out = super().forward(fixed_x.unsqueeze(1))
        out = self.fix_output_dims(out, x)
        return out

    def fix_input_dims(self, x):
        """Overwrite this in subclasses to implement input dimension checks."""
        return x

    def fix_output_dims(self, y, x):
        """Overwrite this in subclasses to implement output dimension checks.
        y is the output and x was the input (passed to use the shape)."""
        return y
