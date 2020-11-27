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
import warnings
from asteroid_filterbanks import transforms
from .utils.torch_utils import script_if_tracing
from .utils.deprecation_utils import mark_deprecated

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import torchaudio
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


@mark_deprecated(
    "Use `torch.view_as_complex`, `torch_complex_from_magphase`, `torch_complex_from_reim` or "
    "`asteroid_filterbanks.transforms.from_torch_complex` instead."
)
@script_if_tracing
def as_torch_complex(x, asteroid_dim: int = -2):
    """Convert complex `x` to complex. Input may be one of:

    - PyTorch native complex
    - Torchaudio style complex
    - Asteroid style complex
    - Tuple or list of (real, imaginary) components

    Args:
        asteroid_dim (int, optional): Dimension to check for Asteroid-style complex.

    Raises:
        ValueError: If type of `x` is not understood.
    """
    if isinstance(x, (list, tuple)) and len(x) == 2:
        return torch_complex_from_reim(*x)
    elif is_torch_complex(x):
        return x
    else:
        is_torchaudio_complex = transforms.is_torchaudio_complex(x)
        is_asteroid_complex = transforms.is_asteroid_complex(x, asteroid_dim)
        if is_torchaudio_complex and is_asteroid_complex:
            raise RuntimeError(
                f"Tensor of shape {x.shape} is both a valid Torchaudio-style and "
                "Asteroid-style complex. PyTorch complex conversion is ambiguous."
            )
        elif is_torchaudio_complex:
            return torch.view_as_complex(x)
        elif is_asteroid_complex:
            return torch.view_as_complex(transforms.to_torchaudio(x, asteroid_dim))
        else:
            raise RuntimeError(
                f"Do not know how to convert tensor of shape {x.shape}, dtype={x.dtype} to complex"
            )


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

    Valid bound types, for a complex mask $M = |M| ⋅ e^{i φ(M)}$:

    - Unbounded ("UBD"): :math:`M_{\mathrm{UBD}} = M`
    - Sigmoid ("BDSS"): :math:`M_{\mathrm{BDSS}} = σ(|M|) e^{i σ(φ(M))}`
    - Tanh ("BDT"): :math:`M_{\mathrm{BDT}} = \mathrm{tanh}(|M|) e^{i φ(M)}`

    Args:
        bound_type (str or None): The type of bound to use, either of
            "tanh"/"bdt" (default), "sigmoid"/"bdss" or None/"bdt".

    References
        - [1] : "Phase-aware Speech Enhancement with Deep Complex U-Net",
          Hyeong-Seok Choi et al. https://arxiv.org/abs/1903.03107
    """
    if bound_type in {"BDSS", "sigmoid"}:
        return on_reim(torch.sigmoid)(mask)
    elif bound_type in {"BDT", "tanh", "UBD", None}:
        mask_mag, mask_phase = torchaudio.functional.magphase(torch.view_as_real(mask))
        if bound_type in {"BDT", "tanh"}:
            mask_mag_bounded = torch.tanh(mask_mag)
        else:
            mask_mag_bounded = mask_mag
        return torch_complex_from_magphase(mask_mag_bounded, mask_phase)
    else:
        raise ValueError(f"Unknown mask bound {bound_type}")
