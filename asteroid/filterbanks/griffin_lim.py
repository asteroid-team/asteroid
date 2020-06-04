import torch
import math

from . import Encoder, Decoder, STFTFB
from .stft_fb import perfect_synthesis_window
from . import transforms
from ..masknn.consistency import mixture_consistency


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
        >>> stft = Encoder(STFTFB(n_filters=256, kernel_size=256, stride=128))
        >>> wav = torch.randn(2, 1, 8000)
        >>> spec = stft(wav)
        >>> masked_spec = spec * torch.sigmoid(torch.randn_like(spec))
        >>> mag = transforms.take_mag(masked_spec, -2)
        >>> est_wav = griffin_lim(mag, stft, n_iter=32)

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


def misi(mixture_wav, mag_specgrams, stft_enc, angles=None, istft_dec=None,
         n_iter=6, momentum=0., src_weights=None, dim=1):
    """ Jointly estimates matching phase from magnitude spectograms using the
    Multiple Input Spectrogram Inversion (MISI) algorithm [1].

    Args:
        mixture_wav (torch.Tensor): (batch, time)
        mag_specgrams (torch.Tensor): (batch, n_src, freq, frames) as
            returned by `Encoder(STFTFB)`, the magnitude spectrograms to be
            jointly inverted using MISI (modified or not).
        stft_enc (Encoder[STFTFB]): The `Encoder(STFTFB())` object that was
            used to compute the input `mag_spec`.
        angles (None or Tensor): Angles to use to initialize the algorithm.
            If None (default), angles are init with uniform ditribution.
        istft_dec (None or Decoder[STFTFB]): Optional Decoder to use to get
            back to the time domain. If None (default), a perfect
            reconstruction Decoder is built from `stft_enc`.
        n_iter (int): Number of MISI iterations to run.
        momentum (float): Momentum on updates (this argument comes from
            GriffinLim). Defaults to 0 as it was never proposed anywhere.
        src_weights (None or torch.Tensor): Consistency weight for each source.
            Shape needs to be broadcastable to `istft_dec(mag_specgrams)`.
            We make sure that the weights sum up to 1 along dim `dim`.
            If `src_weights` is None, compute them based on relative power.
        dim (int): Axis which contains the sources in `mag_specgrams`.
            Used for consistency constraint.

    Returns:
        torch.Tensor: estimated waveforms of shape (batch, n_src, time).

    Examples:
        >>> stft = Encoder(STFTFB(n_filters=256, kernel_size=256, stride=128))
        >>> wav = torch.randn(2, 3, 8000)
        >>> specs = stft(wav)
        >>> masked_specs = specs * torch.sigmoid(torch.randn_like(specs))
        >>> mag = transforms.take_mag(masked_specs, -2)
        >>> est_wav = misi(wav.sum(1), mag, stft, n_iter=32)

    References:
        [1] Gunawan and Sen, "Iterative Phase Estimation for the Synthesis of
        Separated Sources From Single-Channel Mixtures," in IEEE Signal
        Processing Letters, 2010.
        [2] Wang, LeRoux et al. “End-to-End Speech Separation with Unfolded
        Iterative Phase Reconstruction.” Interspeech 2018 (2018)
    """
    # We can create perfect iSTFT from STFT Encoder
    if istft_dec is None:
        # Compute window for perfect resynthesis
        syn_win = perfect_synthesis_window(stft_enc.filterbank.window,
                                           stft_enc.stride)
        istft_dec = Decoder(STFTFB(**stft_enc.get_config(), window=syn_win))

    # If no intitial phase is provided initialize uniformly
    if angles is None:
        angles = 2 * math.pi * torch.rand_like(mag_specgrams,
                                               device=mag_specgrams.device)
    # wav_dim is used in mixture_consistency.
    # Transform spec src dim to wav src dim for positive and negative dim
    wav_dim = dim if dim >= 0 else dim + 1

    # We forward/backward the mixture through STFT to have matching shapes
    # with the input spectrograms as well as  account for potential modulations
    # if the window were not chosen to enable perfect reconstruction.
    mixture_wav = istft_dec(stft_enc(mixture_wav))

    # Initialize rebuilt (useful to use momentum)
    rebuilt = 0.
    for _ in range(n_iter):
        prev_built = rebuilt
        # Go to the time domain
        complex_specgram = transforms.from_mag_and_phase(mag_specgrams, angles)
        wavs = istft_dec(complex_specgram)
        # Make wavs sum up to the mixture
        consistent_wavs = mixture_consistency(mixture_wav, wavs,
                                              src_weights=src_weights,
                                              dim=wav_dim)
        # Back to TF domain
        rebuilt = stft_enc(consistent_wavs)
        # Update phase estimates (with momentum). Keep the momentum here
        # in case. Was shown useful in GF, might be here. We'll see.
        diff = rebuilt - momentum / (1 + momentum) * prev_built
        angles = transforms.angle(diff)
    # Final source estimates
    final_complex_spec = transforms.from_mag_and_phase(mag_specgrams, angles)
    return istft_dec(final_complex_spec)
