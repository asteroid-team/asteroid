import torch
from torch import nn


class SCM(nn.Module):
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, normalize: bool = True):
        """See :func:`compute_scm`."""
        return compute_scm(x, mask=mask, normalize=normalize)


class Beamformer(nn.Module):
    @staticmethod
    def apply_beamforming_vector(bf_vector: torch.Tensor, mix: torch.Tensor):
        """Apply the beamforming vector to the mixture. Output (batch, freqs, frames).

        Args:
            bf_vector: shape (batch, mics, freqs)
            mix: shape (batch, mics, freqs, frames).
        """
        return torch.einsum("...mf,...mft->...ft", bf_vector.conj(), mix)


class RTFMVDRBeamformer(Beamformer):
    def forward(
        self,
        mix: torch.Tensor,
        target_scm: torch.Tensor,
        noise_scm: torch.Tensor,
    ):
        r"""Compute and apply MVDR beamformer from the speech and noise SCM matrices.

        :math:`\mathbf{w} =  \displaystyle \frac{\Sigma_{nn}^{-1} \mathbf{a}}{
        \mathbf{a}^H \Sigma_{nn}^{-1} \mathbf{a}}` where :math:`\mathbf{a}` is the ATF estimated from the target SCM.

        Args:
            mix (torch.ComplexTensor): shape (batch, mics, freqs, frames)
            target_scm (torch.ComplexTensor): (batch, mics, mics, freqs)
            noise_scm (torch.ComplexTensor): (batch, mics, mics, freqs)

        Returns:
            Filtered mixture. torch.ComplexTensor (batch, freqs, frames)
        """
        # TODO: Implement several RTF estimation strategies, and choose one here, or expose all.
        # Get relative transfer function (1st PCA of Σss)
        e_val, e_vec = torch.symeig(target_scm.permute(0, 3, 1, 2), eigenvectors=True)
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
    # TODO(popcornell): fill in the code.
    pass


class SDWMWFBeamformer(Beamformer):
    def __init__(self, mu=1.0):
        super().__init__()
        self.mu = mu

    def forward(
        self, mix: torch.Tensor, target_scm: torch.Tensor, noise_scm: torch.Tensor, ref_mic: int = 0
    ):
        """Compute and apply SDW-MWF beamformer.

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

        denominator = target_scm_t + self.mu * noise_scm_t
        bf_vect = stable_solve(target_scm_t, denominator)
        bf_vect = bf_vect[..., ref_mic].transpose(-1, -2)  # -> bfm1  -> bmf
        output = self.apply_beamforming_vector(bf_vect, mix=mix)  # -> bft
        return output


class GEVBeamformer(Beamformer):
    def forward(self, mix: torch.Tensor, target_scm: torch.Tensor, noise_scm: torch.Tensor):
        """Compute and apply the GEV beamformer.

        :math:`\mathbf{w} =  \displaystyle MaxEig\{ \Sigma_{nn}^{-1}\Sigma_{ss} \}`, where
        MaxEig extracts the eigenvector corresponding to the maximum eigenvalue (using the GEV decomposition).

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


def condition_scm(x, eps=1e-6, dim1=-2, dim2=-1):
    """Condition input SCM with (x + eps tr(x) I) / (1 + eps) along `dim1` and `dim2`.

    See https://stt.msu.edu/users/mauryaas/Ashwini_JPEN.pdf (2.3).
    """
    # Assume 4d with ...mm
    if dim1 != -2 or dim2 != -1:
        raise NotImplementedError
    scale = eps * batch_trace(x, dim1=dim1, dim2=dim2)[..., None, None] / x.shape[dim1]
    scaled_eye = torch.eye(x.shape[dim1])[None, None] * scale
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
        solve_dtype = _to_double_map[input_dtype]
    return _stable_solve(b.to(solve_dtype), a.to(solve_dtype)).to(input_dtype)


def _stable_solve(b, a, eps=1e-6):
    try:
        return torch.solve(b, a)[0]
    except RuntimeError:
        a = condition_scm(a, eps)
        return torch.solve(b, a)[0]


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
        solve_dtype = _to_double_map[input_dtype]
    return _stable_cholesky(input.to(solve_dtype), upper=upper, out=out, eps=eps).to(input_dtype)


def _stable_cholesky(input, upper=False, out=None, eps=1e-6):
    try:
        return torch.cholesky(input, upper=upper, out=out)
    except RuntimeError:
        input = condition_scm(input, eps)
        return torch.cholesky(input, upper=upper, out=out)


def generalized_eigenvalue_decomposition(a, b):
    """Solves the generalized eigenvalue decomposition through Cholesky decomposition.
    Returns eigen values and eigen vectors (ascending order).
    """
    # Only run it in double
    input_dtype = _common_dtype(a, b)
    solve_dtype = input_dtype
    if input_dtype not in [torch.float64, torch.complex128]:
        solve_dtype = _to_double_map[input_dtype]
    e_val, e_vec = _generalized_eigenvalue_decomposition(a.to(solve_dtype), b.to(solve_dtype))
    return e_val.to(input_dtype), e_vec.to(input_dtype)


def _generalized_eigenvalue_decomposition(a, b):
    cholesky = stable_cholesky(b)
    inv_cholesky = torch.inverse(cholesky)
    # Compute C matrix L⁻1 A L^-T
    cmat = inv_cholesky @ a @ inv_cholesky.conj().transpose(-1, -2)
    # Performing the eigenvalue decomposition
    e_val, e_vec = torch.symeig(cmat, eigenvectors=True)
    # Collecting the eigenvectors
    e_vec = torch.matmul(inv_cholesky.conj().transpose(-1, -2), e_vec)
    return e_val, e_vec


_to_double_map = {
    torch.float16: torch.float64,
    torch.float32: torch.float64,
    torch.complex32: torch.complex128,
    torch.complex64: torch.complex128,
}


def _common_dtype(*args):
    all_dtypes = [a.dtype for a in args]
    if len(set(all_dtypes)) > 1:
        raise RuntimeError(f"Expected inputs from the same dtype. Received {all_dtypes}.")
    return all_dtypes[0]


# Legacy
BeamFormer = Beamformer
SdwMwfBeamformer = SDWMWFBeamformer
MvdrBeamformer = RTFMVDRBeamformer
