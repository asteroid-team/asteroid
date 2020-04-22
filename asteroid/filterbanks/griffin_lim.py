import torch
import math

from . import Encoder, Decoder, STFTFB
from .stft_fb import perfect_synthesis_window
from . import transforms


def griffin_lim(mag_specgram, stft_enc, angles=None, istft_dec=None, n_iter=6,
                momentum=0.9):
    """ Estimates matching phase from magnitude spectogram using the
    'fast' Griffin Lim algorithm [1].

    Args:
        mag_specgram (torch.Tensor): (any, dim, ension, freq, frames) as
            returned by `Encoder(STFTFB)`, the magnitude spectrogram to be
            inverted.
        stft_enc (Encoder[STFTFB]): The `Encoder(STFTFB())` object that was
            used to compute the input `mag_spec`.
        angles (None or Tensor): Angles to use to initialize the algorithm.
            If None (default), angles are init with uniform ditribution.
        istft_dec (None or Decoder[STFTFB]): Optional Decoder to use to get
            back to the time domain. If None (default), a perfect
            reconstruction Decoder is built from `stft_enc`.
        n_iter (int): Number of griffin-lim iterations to run.
        momentum (float): The momentum of fast Griffin-Lim. Original
            Griffin-Lim is obtained for momentum=0.

    Returns:
        torch.Tensor: estimated waveforms of shape (any, dim, ension, time).

    Examples:
        To come.

    References:
        [1] Perraudin et al. "A fast Griffin-Lim algorithm," WASPAA 2013.
        [2] D. W. Griffin and J. S. Lim:  "Signal estimation from modified
        short-time Fourier transform," ASSP 1984.

    """
    # We can create perfect iSTFT from STFT Encoder
    if istft_dec is None:
        # Compute window for perfect resynthesis
        syn_win = perfect_synthesis_window(stft_enc.filterbank.window,
                                           stft_enc.stride)
        istft_dec = Decoder(STFTFB(**stft_enc.get_config(), window=syn_win))

    # If no intitial phase is provided initialize uniformly
    if angles is None:
        angles = 2 * math.pi * torch.rand_like(mag_specgram,
                                               device=mag_specgram.device)
    else:
        angles = angles.view(*mag_specgram.shape)

    # Initialize rebuilt (useful to use momentum)
    rebuilt = 0.
    for _ in range(n_iter):
        prev_built = rebuilt
        # Go to the time domain
        complex_specgram = transforms.from_mag_and_phase(mag_specgram, angles)
        waveform = istft_dec(complex_specgram)
        # And back to TF domain
        rebuilt = stft_enc(waveform)
        # Update phase estimates (with momentum)
        diff = rebuilt - momentum / (1 + momentum) * prev_built
        angles = transforms.angle(diff)

    final_complex_spec = transforms.from_mag_and_phase(mag_specgram, angles)
    return istft_dec(final_complex_spec)
