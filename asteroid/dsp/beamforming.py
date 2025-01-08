from typing import Union
import torch
from torch import nn
from torch.nn import functional as F


class SCM(nn.Module):
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, normalize: bool = True):
        """See :func:`compute_scm`."""
        return compute_scm(x, mask=mask, normalize=normalize)


class Beamformer(nn.Module):
    """Base class for beamforming modules."""

    @staticmethod
    def apply_beamforming_vector(bf_vector: torch.Tensor, mix: torch.Tensor):
        """Apply the beamforming vector to the mixture. Output (batch, freqs, frames).

        Args:
            bf_vector: shape (batch, mics, freqs)
            mix: shape (batch, mics, freqs, frames).
        """
        return torch.einsum("...mf,...mft->...ft", bf_vector.conj(), mix)

    @staticmethod
    def get_reference_mic_vects(
        ref_mic,
        bf_mat: torch.Tensor,
        target_scm: torch.Tensor = None,
        noise_scm: torch.Tensor = None,
    ):
        """Return the reference channel indices over the batch.

        Args:
            ref_mic (Optional[Union[int, torch.Tensor]]): The reference channel.
                If torch.Tensor (ndim>1), return it, it is the reference mic vector,
                If torch.LongTensor of size `batch`, select independent reference mic of the batch.
                If int, select the corresponding reference mic,
                If None, the optimal reference mics are computed with :func:`get_optimal_reference_mic`,
                If None, and either SCM is None, `ref_mic` is set to `0`,
            bf_mat: beamforming matrix of shape (batch, freq, mics, mics).
            target_scm (torch.ComplexTensor): (batch, freqs, mics, mics).
            noise_scm (torch.ComplexTensor): (batch, freqs, mics, mics).

        Returns:
            torch.LongTensor of size ``batch`` to select with the reference channel indices.
        """
        # If ref_mic already has the expected shape.
        if isinstance(ref_mic, torch.Tensor) and ref_mic.ndim > 1:
            return ref_mic

        if (target_scm is None or noise_scm is None) and ref_mic is None:
            ref_mic = 0
        if ref_mic is None:
            batch_mic_idx = get_optimal_reference_mic(
                bf_mat=bf_mat, target_scm=target_scm, noise_scm=noise_scm
            )
        elif isinstance(ref_mic, int):
            batch_mic_idx = torch.LongTensor([ref_mic] * bf_mat.shape[0]).to(bf_mat.device)
        elif isinstance(ref_mic, torch.Tensor):  # Must be 1D
            batch_mic_idx = ref_mic
        else:
            raise ValueError(
                f"Unsupported reference microphone format. Support None, int and 1D "
                f"torch.LongTensor and torch.Tensor, received {type(ref_mic)}."
            )
        # Output (batch, 1, n_mics, 1)
        # import ipdb; ipdb.set_trace()
        ref_mic_vects = F.one_hot(batch_mic_idx, num_classes=bf_mat.shape[-1])[:, None, :, None]
        return ref_mic_vects.to(bf_mat.dtype).to(bf_mat.device)


class RTFMVDRBeamformer(Beamformer):
    def forward(
        self,
        mix: torch.Tensor,
        target_scm: torch.Tensor,
        noise_scm: torch.Tensor,
    ):
        r"""Compute and apply MVDR beamformer from the speech and noise SCM matrices.

        :math:`\mathbf{w} =  \displaystyle \frac{\Sigma_{nn}^{-1} \mathbf{a}}{
        \mathbf{a}^H \Sigma_{nn}^{-1} \mathbf{a}}` where :math:`\mathbf{a}` is the
        ATF estimated from the target SCM.

        Args:
            mix (torch.ComplexTensor): shape (batch, mics, freqs, frames)
            target_scm (torch.ComplexTensor): (batch, mics, mics, freqs)
            noise_scm (torch.ComplexTensor): (batch, mics, mics, freqs)

        Returns:
            Filtered mixture. torch.ComplexTensor (batch, freqs, frames)
        """
        # TODO: Implement several RTF estimation strategies, and choose one here, or expose all.
        # Get relative transfer function (1st PCA of Σss)
        e_val, e_vec = torch.linalg.eigh(target_scm.permute(0, 3, 1, 2))
        rtf_vect = e_vec[..., -1]  # bfm
        return self.from_rtf_vect(mix=mix, rtf_vec=rtf_vect.transpose(-1, -2), noise_scm=noise_scm)

    def from_rtf_vect(
        self,
        mix: torch.Tensor,
        rtf_vec: torch.Tensor,
        noise_scm: torch.Tensor,
    ):
        """Compute and apply MVDR beamformer from the ATF vector and noise SCM matrix.

        Args:
            mix (torch.ComplexTensor): shape (batch, mics, freqs, frames)
            rtf_vec (torch.ComplexTensor): (batch, mics, freqs)
            noise_scm (torch.ComplexTensor): (batch, mics, mics, freqs)

        Returns:
            Filtered mixture. torch.ComplexTensor (batch, freqs, frames)
        """
        noise_scm_t = noise_scm.permute(0, 3, 1, 2)  # -> bfmm
        rtf_vec_t = rtf_vec.transpose(-1, -2).unsqueeze(-1)  # -> bfm1

        numerator = stable_solve(rtf_vec_t, noise_scm_t)  # -> bfm1

        denominator = torch.matmul(rtf_vec_t.conj().transpose(-1, -2), numerator)  # -> bf11
        bf_vect = (numerator / denominator).squeeze(-1).transpose(-1, -2)  # -> bfm1  -> bmf
        output = self.apply_beamforming_vector(bf_vect, mix=mix)  # -> bft
        return output


class SoudenMVDRBeamformer(Beamformer):
    def forward(
        self,
        mix: torch.Tensor,
        target_scm: torch.Tensor,
        noise_scm: torch.Tensor,
        ref_mic: Union[torch.Tensor, torch.LongTensor, int] = 0,
        eps=1e-8,
    ):
        r"""Compute and apply MVDR beamformer from the speech and noise SCM matrices.
        This class uses Souden's formulation [1].

        :math:`\mathbf{w} =  \displaystyle \frac{\Sigma_{nn}^{-1} \Sigma_{ss}}{
        Tr\left( \Sigma_{nn}^{-1} \Sigma_{ss} \right) }\mathbf{u}` where :math:`\mathbf{a}`
        is the steering vector.


        Args:
            mix (torch.ComplexTensor): shape (batch, mics, freqs, frames)
            target_scm (torch.ComplexTensor): (batch, mics, mics, freqs)
            noise_scm (torch.ComplexTensor): (batch, mics, mics, freqs)
            ref_mic (int): reference microphone.
            eps: numerical stabilizer.

        Returns:
            Filtered mixture. torch.ComplexTensor (batch, freqs, frames)

        References
            [1] Souden, M., Benesty, J., & Affes, S. (2009). On optimal frequency-domain multichannel
            linear filtering for noise reduction. IEEE Transactions on audio, speech, and language processing, 18(2), 260-276.
        """
        noise_scm = noise_scm.permute(0, 3, 1, 2)  # -> bfmm
        target_scm = target_scm.permute(0, 3, 1, 2)  # -> bfmm

        numerator = stable_solve(target_scm, noise_scm)
        bf_mat = numerator / (batch_trace(numerator)[..., None, None] + eps)  # bfmm

        # allow for a-posteriori SNR selection
        batch_mic_vects = self.get_reference_mic_vects(
            ref_mic, bf_mat, target_scm=target_scm, noise_scm=noise_scm
        )
        bf_vect = torch.matmul(bf_mat, batch_mic_vects)  # -> bfmm  -> bfm1
        bf_vect = bf_vect.squeeze(-1).transpose(-1, -2)  # bfm1 -> bmf
        output = self.apply_beamforming_vector(bf_vect, mix=mix)  # -> bft
        return output


class SDWMWFBeamformer(Beamformer):
    def __init__(self, mu=1.0):
        super().__init__()
        self.mu = mu

    def forward(
        self,
        mix: torch.Tensor,
        target_scm: torch.Tensor,
        noise_scm: torch.Tensor,
        ref_mic: Union[torch.Tensor, torch.LongTensor, int] = None,
    ):
        r"""Compute and apply SDW-MWF beamformer.

        :math:`\mathbf{w} =  \displaystyle (\Sigma_{ss} + \mu \Sigma_{nn})^{-1} \Sigma_{ss}`.

        Args:
            mix (torch.ComplexTensor): shape (batch, mics, freqs, frames)
            target_scm (torch.ComplexTensor): (batch, mics, mics, freqs)
            noise_scm (torch.ComplexTensor): (batch, mics, mics, freqs)
            ref_mic (int): reference microphone.

        Returns:
            Filtered mixture. torch.ComplexTensor (batch, freqs, frames)
        """
        noise_scm_t = noise_scm.permute(0, 3, 1, 2)  # -> bfmm
        target_scm_t = target_scm.permute(0, 3, 1, 2)  # -> bfmm

        # import ipdb; ipdb.set_trace()

        denominator = target_scm_t + self.mu * noise_scm_t
        bf_mat = stable_solve(target_scm_t, denominator)
        # Reference mic selection and application
        batch_mic_vects = self.get_reference_mic_vects(
            ref_mic, bf_mat, target_scm=target_scm_t, noise_scm=noise_scm_t
        )  # b1m1
        bf_vect = torch.matmul(bf_mat, batch_mic_vects)  # -> bfmm  -> bfm1
        bf_vect = bf_vect.squeeze(-1).transpose(-1, -2)  # bfm1 -> bmf
        output = self.apply_beamforming_vector(bf_vect, mix=mix)  # -> bft
        return output


class GEVBeamformer(Beamformer):
    def forward(self, mix: torch.Tensor, target_scm: torch.Tensor, noise_scm: torch.Tensor):
        r"""Compute and apply the GEV beamformer.

        :math:`\mathbf{w} =  \displaystyle MaxEig\{ \Sigma_{nn}^{-1}\Sigma_{ss} \}`, where
        MaxEig extracts the eigenvector corresponding to the maximum eigenvalue
        (using the GEV decomposition).

        Args:
            mix: shape (batch, mics, freqs, frames)
            target_scm: (batch, mics, mics, freqs)
            noise_scm: (batch, mics, mics, freqs)

        Returns:
            Filtered mixture. torch.ComplexTensor (batch, freqs, frames)
        """
        bf_vect = self.compute_beamforming_vector(target_scm, noise_scm)
        output = self.apply_beamforming_vector(bf_vect, mix=mix)  # -> bft
        return output

    @staticmethod
    def compute_beamforming_vector(target_scm: torch.Tensor, noise_scm: torch.Tensor):
        noise_scm_t = noise_scm.permute(0, 3, 1, 2)
        noise_scm_t = condition_scm(noise_scm_t, 1e-6)
        e_val, e_vec = generalized_eigenvalue_decomposition(
            target_scm.permute(0, 3, 1, 2), noise_scm_t
        )
        bf_vect = e_vec[..., -1]
        # Normalize
        bf_vect /= torch.norm(bf_vect, dim=-1, keepdim=True)
        bf_vect = bf_vect.squeeze(-1).transpose(-1, -2)  # -> bft
        return bf_vect


class GEVDBeamformer(Beamformer):
    """Generalized eigenvalue decomposition speech distortion weighted multichannel Wiener filter.

        Compare to SDW-MWF, spatial covariance matrix are computed from low rank approximation
        based on eigen values decomposition,
        see equation 62 in `[1] <https://hal.inria.fr/hal-01390918/file/14-1.pdf>`_.

    Attributes:
        mu (float): Speech distortion constant.
        rank (int): Rank for the approximation of target covariance matrix,
            no approximation is made if `rank` is None.

    References:
        [1] R. Serizel, M. Moonen, B. Van Dijk and J. Wouters,
        "Low-rank Approximation Based Multichannel Wiener Filter Algorithms for
        Noise Reduction with Application in Cochlear Implants,"
        in IEEE/ACM Transactions on Audio, Speech, and Language Processing, April 2014.
    """

    def __init__(self, mu: float = 1.0, rank: int = 1):
        super().__init__()
        self.mu = mu
        self.rank = rank

    def compute_beamforming_vector(self, target_scm: torch.Tensor, noise_scm: torch.Tensor):
        """Compute beamforming vectors for GEVD beamFormer.

        Args:
            target_scm (torch.ComplexTensor): shape (batch, mics, mics, freqs)
            noise_scm (torch.ComplexTensor): shape (batch, mics, mics, freqs)

        Returns:
            torch.ComplexTensor: shape (batch, mics, freqs)

        """
        #  GEV decomposition of noise_scm^(-1) * target_scm
        e_values, e_vectors = _generalized_eigenvalue_decomposition(
            target_scm.permute(0, 3, 1, 2),  # bmmf -> bfmm
            noise_scm.permute(0, 3, 1, 2),  # bmmf -> bfmm
        )

        #  Prevent negative and infinite eigenvalues
        eps = torch.finfo(e_values.dtype).eps
        e_values = torch.clamp(e_values, min=eps, max=1e6)

        #  Sort eigen values and vectors in descending-order
        e_values = torch.diag_embed(torch.flip(e_values, [-1]))
        e_vectors = torch.flip(e_vectors, [-1])

        #  Force zero values for all GEV but the highest
        if self.rank:
            e_values[..., self.rank :, :] = 0.0

        #  Compute bf vectors as SDW MWF filter  in eigen space
        complex_type = e_vectors.dtype
        ev_plus_mu = e_values + self.mu * torch.eye(e_values.shape[-1]).expand_as(e_values)
        bf_vect = (
            e_vectors
            @ e_values.to(complex_type)
            @ torch.linalg.inv(e_vectors @ ev_plus_mu.to(complex_type))
        )

        return bf_vect[..., 0].permute(0, 2, 1)  # bfmm -> bfm -> bmf

    def forward(
        self,
        mix: torch.Tensor,
        target_scm: torch.Tensor,
        noise_scm: torch.Tensor,
    ):
        """Compute and apply the GEVD beamformer.

        Args:
            mix (torch.ComplexTensor): shape (batch, mics, freqs, frames)
            target_scm (torch.ComplexTensor): (batch, mics, mics, freqs)
            noise_scm (torch.ComplexTensor): (batch, mics, mics, freqs)

        Returns:
            Filtered mixture. torch.ComplexTensor (batch, freqs, frames)
        """
        bf_vect = self.compute_beamforming_vector(target_scm, noise_scm)
        return self.apply_beamforming_vector(bf_vect, mix=mix)


def compute_scm(x: torch.Tensor, mask: torch.Tensor = None, normalize: bool = True):
    """Compute the spatial covariance matrix from a STFT signal x.

    Args:
        x (torch.ComplexTensor): shape  [batch, mics, freqs, frames]
        mask (torch.Tensor): [batch, 1, freqs, frames] or [batch, 1, freqs, frames]. Optional
        normalize (bool): Whether to normalize with the mask mean per bin.

    Returns:
        torch.ComplexTensor, the SCM with shape (batch, mics, mics, freqs)
    """
    batch, mics, freqs, frames = x.shape
    if mask is None:
        mask = torch.ones(batch, 1, freqs, frames)
    if mask.ndim == 3:
        mask = mask[:, None]

    # torch.matmul((mask * x).transpose(1, 2), x.conj().permute(0, 2, 3, 1))
    scm = torch.einsum("bmft,bnft->bmnf", mask * x, x.conj())
    if normalize:
        scm /= mask.sum(-1, keepdim=True).transpose(-1, -2)
    return scm


def get_optimal_reference_mic(
    bf_mat: torch.Tensor,
    target_scm: torch.Tensor,
    noise_scm: torch.Tensor,
    eps: float = 1e-6,
):
    """Compute the optimal reference mic given the a posteriori SNR, see [1].

    Args:
        bf_mat: (batch, freq, mics, mics)
        target_scm (torch.ComplexTensor): (batch, freqs, mics, mics)
        noise_scm (torch.ComplexTensor): (batch, freqs, mics, mics)
        eps: value to clip the denominator.

    Returns:
        torch.

    References
        Erdogan et al. 2016: "Improved MVDR beamforming using single-channel maskprediction networks"
            https://www.merl.com/publications/docs/TR2016-072.pdf
    """
    den = torch.clamp(
        torch.einsum("...flm,...fln,...fnm->...m", bf_mat.conj(), noise_scm, bf_mat).real, min=eps
    )
    snr_post = (
        torch.einsum("...flm,...fln,...fnm->...m", bf_mat.conj(), target_scm, bf_mat).real / den
    )
    assert torch.all(torch.isfinite(snr_post)), snr_post
    return torch.argmax(snr_post, dim=-1)


def condition_scm(x, eps=1e-6, dim1=-2, dim2=-1):
    """Condition input SCM with (x + eps tr(x) I) / (1 + eps) along `dim1` and `dim2`.

    See https://stt.msu.edu/users/mauryaas/Ashwini_JPEN.pdf (2.3).
    """
    # Assume 4d with ...mm
    if dim1 != -2 or dim2 != -1:
        raise NotImplementedError
    scale = eps * batch_trace(x, dim1=dim1, dim2=dim2)[..., None, None] / x.shape[dim1]
    scaled_eye = torch.eye(x.shape[dim1], device=x.device)[None, None] * scale
    return (x + scaled_eye) / (1 + eps)


def batch_trace(x, dim1=-2, dim2=-1):
    """Compute the trace along `dim1` and `dim2` for a any matrix `ndim>=2`."""
    return torch.diagonal(x, dim1=dim1, dim2=dim2).sum(-1)


def stable_solve(b, a):
    """Return torch.solve if `a` is non-singular, else regularize `a` and return torch.solve."""
    # Only run it in double
    input_dtype = _common_dtype(b, a)
    solve_dtype = input_dtype
    if input_dtype not in [torch.float64, torch.complex128]:
        solve_dtype = _precision_mapping()[input_dtype]
    return _stable_solve(b.to(solve_dtype), a.to(solve_dtype)).to(input_dtype)


def _stable_solve(b, a, eps=1e-6):
    try:
        return torch.linalg.solve(a, b)
    except RuntimeError:
        a = condition_scm(a, eps)
        return torch.linalg.solve(a, b)


def stable_cholesky(input, upper=False, out=None, eps=1e-6):
    """Compute the Cholesky decomposition of ``input``.
    If ``input`` is only p.s.d, add a small jitter to the diagonal.

    Args:
        input (Tensor): The tensor to compute the Cholesky decomposition of
        upper (bool, optional): See torch.cholesky
        out (Tensor, optional): See torch.cholesky
        eps (int): small jitter added to the diagonal if PD.
    """
    # Only run it in double
    input_dtype = input.dtype
    solve_dtype = input_dtype
    if input_dtype not in [torch.float64, torch.complex128]:
        solve_dtype = _precision_mapping()[input_dtype]
    return _stable_cholesky(input.to(solve_dtype), upper=upper, out=out, eps=eps).to(input_dtype)


def _stable_cholesky(input, upper=False, out=None, eps=1e-6):
    try:
        if upper:
            return torch.linalg.cholesky(input, out=out).mH
        return torch.linalg.cholesky(input, out=out)
    except RuntimeError:
        input = condition_scm(input, eps)
        if upper:
            return torch.linalg.cholesky(input, out=out).mH
        return torch.linalg.cholesky(input, out=out)


def generalized_eigenvalue_decomposition(a, b):
    """Solves the generalized eigenvalue decomposition through Cholesky decomposition.
    Returns eigen values and eigen vectors (ascending order).
    """
    # Only run it in double
    input_dtype = _common_dtype(a, b)
    solve_dtype = input_dtype
    if input_dtype not in [torch.float64, torch.complex128]:
        solve_dtype = _precision_mapping()[input_dtype]
    e_val, e_vec = _generalized_eigenvalue_decomposition(a.to(solve_dtype), b.to(solve_dtype))
    return e_val.to(input_dtype).real, e_vec.to(input_dtype)


def _generalized_eigenvalue_decomposition(a, b):
    cholesky = stable_cholesky(b)
    inv_cholesky = torch.inverse(cholesky)
    # Compute C matrix L⁻1 A L^-T
    cmat = inv_cholesky @ a @ inv_cholesky.conj().transpose(-1, -2)
    # Performing the eigenvalue decomposition
    e_val, e_vec = torch.linalg.eigh(cmat)
    # Collecting the eigenvectors
    e_vec = torch.matmul(inv_cholesky.conj().transpose(-1, -2), e_vec)
    return e_val, e_vec


def _common_dtype(*args):
    all_dtypes = [a.dtype for a in args]
    if len(set(all_dtypes)) > 1:
        raise RuntimeError(f"Expected inputs from the same dtype. Received {all_dtypes}.")
    return all_dtypes[0]


USE_DOUBLE = True


def force_float_linalg():
    global USE_DOUBLE
    USE_DOUBLE = False


def force_double_linalg():
    global USE_DOUBLE
    USE_DOUBLE = True


def _precision_mapping():
    has_complex32 = hasattr(torch, "complex32")
    if USE_DOUBLE:
        precision_map = {
            torch.float16: torch.float64,
            torch.float32: torch.float64,
            torch.complex64: torch.complex128,
        }
        if has_complex32:
            precision_map[torch.complex32] = torch.complex128
    else:
        precision_map = {
            torch.float16: torch.float16,
            torch.float32: torch.float32,
            torch.complex64: torch.complex64,
        }
        if has_complex32:
            precision_map[torch.complex32] = torch.complex32
    return precision_map


# Legacy
BeamFormer = Beamformer
SdwMwfBeamformer = SDWMWFBeamformer
MvdrBeamformer = RTFMVDRBeamformer
