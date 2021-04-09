"""Complex building blocks that work with PyTorch native (!) complex tensors, i.e.
dtypes complex64/complex128, or tensors for which `.is_complex()` returns True.

Note that Asteroid code has two other representations of complex numbers:

- Torchaudio representation [..., 2] where [..., 0] and [..., 1] are real and
  imaginary components, respectively
- Asteroid style representation, identical to the Torchaudio representation, but
  with the last dimension concatenated: tensor([r1, r2, ..., rn, i1, i2, ..., in]).
  The concatenated (2 * n) dimension may be at an arbitrary position, i.e. the tensor
  is of shape [..., 2 * n, ...].  See `asteroid_filterbanks.transforms` for details.
"""
import functools
import torch
from asteroid_filterbanks import transforms
from torch import nn


# Alias to denote PyTorch native complex tensor (complex64/complex128).
# `.is_complex()` returns True on these tensors.
ComplexTensor = torch.Tensor


def is_torch_complex(x):
    return x.is_complex()


def torch_complex_from_magphase(mag, phase):
    return torch.view_as_complex(
        torch.stack((mag * torch.cos(phase), mag * torch.sin(phase)), dim=-1)
    )


def torch_complex_from_reim(re, im):
    return torch.view_as_complex(torch.stack([re, im], dim=-1))


def on_reim(f):
    """Make a complex-valued function callable from a real-valued one by applying it to
    the real and imaginary components independently.

    Return:
        cf(x), complex version of `f`: A function that applies `f` to the real and
        imaginary components of `x` and returns the result as PyTorch complex tensor.
    """

    @functools.wraps(f)
    def cf(x):
        return torch_complex_from_reim(f(x.real), f(x.imag))

    # functools.wraps keeps the original name of `f`, which might be confusing,
    # since we are creating a new function that behaves differently.
    # Both __name__ and __qualname__ are used by printing code.
    cf.__name__ == f"{f.__name__} (complex)"
    cf.__qualname__ == f"{f.__qualname__} (complex)"
    return cf


class OnReIm(nn.Module):
    """Like `on_reim`, but for stateful modules.

    Args:
        module_cls (callable): A class or function that returns a Torch module/functional.
            Called 2x with *args, **kwargs, to construct the real and imaginary component modules.
    """

    def __init__(self, module_cls, *args, **kwargs):
        super().__init__()
        self.re_module = module_cls(*args, **kwargs)
        self.im_module = module_cls(*args, **kwargs)

    def forward(self, x):
        return torch_complex_from_reim(self.re_module(x.real), self.im_module(x.imag))


class ComplexMultiplicationWrapper(nn.Module):
    """Make a complex-valued module `F` from a real-valued module `f` by applying
    complex multiplication rules:

    F(a + i b) = f1(a) - f1(b) + i (f2(b) + f2(a))

    where `f1`, `f2` are instances of `f` that do *not* share weights.

    Args:
        module_cls (callable): A class or function that returns a Torch module/functional.
            Constructor of `f` in the formula above.  Called 2x with `*args`, `**kwargs`,
            to construct the real and imaginary component modules.
    """

    def __init__(self, module_cls, *args, **kwargs):
        super().__init__()
        self.re_module = module_cls(*args, **kwargs)
        self.im_module = module_cls(*args, **kwargs)

    def forward(self, x: ComplexTensor) -> ComplexTensor:
        return torch_complex_from_reim(
            self.re_module(x.real) - self.im_module(x.imag),
            self.re_module(x.imag) + self.im_module(x.real),
        )


class ComplexSingleRNN(nn.Module):
    """Module for a complex RNN block.

    This is similar to :cls:`asteroid.masknn.recurrent.SingleRNN` but uses complex
    multiplication as described in [1]. Arguments are identical to those of `SingleRNN`,
    except for `dropout`, which is not yet supported.

    Args:
        rnn_type (str): Select from ``'RNN'``, ``'LSTM'``, ``'GRU'``. Can
            also be passed in lowercase letters.
        input_size (int): Dimension of the input feature. The input should have
            shape [batch, seq_len, input_size].
        hidden_size (int): Dimension of the hidden state.
        n_layers (int, optional): Number of layers used in RNN. Default is 1.
        bidirectional (bool, optional): Whether the RNN layers are
            bidirectional. Default is ``False``.
        dropout: Not yet supported.

    References
        [1] : "DCCRN: Deep Complex Convolution Recurrent Network for Phase-Aware Speech Enhancement",
        Yanxin Hu et al. https://arxiv.org/abs/2008.00264
    """

    def __init__(
        self, rnn_type, input_size, hidden_size, n_layers=1, dropout=0, bidirectional=False
    ):
        assert not (dropout and n_layers > 1), "Dropout is not yet supported for complex RNN"
        super().__init__()
        from .masknn.recurrent import SingleRNN  # Avoid circual import

        kwargs = {
            "rnn_type": rnn_type,
            "hidden_size": hidden_size,
            "n_layers": 1,
            "dropout": 0,
            "bidirectional": bidirectional,
        }
        first_rnn = ComplexMultiplicationWrapper(SingleRNN, input_size=input_size, **kwargs)
        self.rnns = torch.nn.ModuleList([first_rnn])
        for _ in range(n_layers - 1):
            self.rnns.append(
                ComplexMultiplicationWrapper(
                    SingleRNN, input_size=first_rnn.re_module.output_size, **kwargs
                )
            )

    @property
    def output_size(self):
        return self.rnns[-1].re_module.output_size

    def forward(self, x: ComplexTensor) -> ComplexTensor:
        """Input shape [batch, seq, feats]"""
        for rnn in self.rnns:
            x = rnn(x)
        return x


ComplexConv2d = functools.partial(ComplexMultiplicationWrapper, nn.Conv2d)
ComplexConvTranspose2d = functools.partial(ComplexMultiplicationWrapper, nn.ConvTranspose2d)


class BoundComplexMask(nn.Module):
    """Module version of `bound_complex_mask`"""

    def __init__(self, bound_type):
        super().__init__()
        self.bound_type = bound_type

    def forward(self, mask: ComplexTensor):
        return bound_complex_mask(mask, self.bound_type)


def bound_complex_mask(mask: ComplexTensor, bound_type="tanh"):
    r"""Bound a complex mask, as proposed in [1], section 3.2.

    Valid bound types, for a complex mask :math:`M = |M| ⋅ e^{i φ(M)}`:

    - Unbounded ("UBD"): :math:`M_{\mathrm{UBD}} = M`
    - Sigmoid ("BDSS"): :math:`M_{\mathrm{BDSS}} = σ(|M|) e^{i σ(φ(M))}`
    - Tanh ("BDT"): :math:`M_{\mathrm{BDT}} = \mathrm{tanh}(|M|) e^{i φ(M)}`

    Args:
        bound_type (str or None): The type of bound to use, either of
            "tanh"/"bdt" (default), "sigmoid"/"bdss" or None/"bdt".

    References
        [1] : "Phase-aware Speech Enhancement with Deep Complex U-Net",
        Hyeong-Seok Choi et al. https://arxiv.org/abs/1903.03107
    """
    if bound_type in {"BDSS", "sigmoid"}:
        return on_reim(torch.sigmoid)(mask)
    elif bound_type in {"BDT", "tanh", "UBD", None}:
        mask_mag, mask_phase = transforms.magphase(transforms.from_torch_complex(mask))
        if bound_type in {"BDT", "tanh"}:
            mask_mag_bounded = torch.tanh(mask_mag)
        else:
            mask_mag_bounded = mask_mag
        return torch_complex_from_magphase(mask_mag_bounded, mask_phase)
    else:
        raise ValueError(f"Unknown mask bound {bound_type}")
