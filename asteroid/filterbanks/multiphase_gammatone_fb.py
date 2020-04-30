import numpy as np
import torch
from .enc_dec import Filterbank


class MultiphaseGammatoneFB(Filterbank):
    """ Multi-Phase Gammatone Filterbank as described in [1].
    Please cite [1] whenever using this.
    Original code repository: `<https://github.com/sp-uhh/mp-gtf>`

    Args:
        n_filters (int): Number of filters.
        kernel_size (int): Length of the filters.
        sample_rate (int, optional): The sample rate (used for initialization).
        stride (int, optional): Stride of the convolution. If None (default),
            set to ``kernel_size // 2``.

    References:
    [1] David Ditter, Timo Gerkmann, "A Multi-Phase Gammatone Filterbank for
        Speech Separation via TasNet", ICASSP 2020
        Available: `<https://ieeexplore.ieee.org/document/9053602/>`
    """
    def __init__(self, n_filters=128, kernel_size=16, sample_rate=8000,
                 stride=None, **kwargs):
        super().__init__(n_filters, kernel_size, stride=stride)
        self.sample_rate = sample_rate
        self.n_feats_out = n_filters
        length_in_seconds = kernel_size / sample_rate
        mpgtf = generate_mpgtf(sample_rate, length_in_seconds, n_filters)
        filters = torch.from_numpy(mpgtf).unsqueeze(1).float()
        self.register_buffer("_filters", filters)

    @property
    def filters(self):
        return self._filters


def generate_mpgtf(samplerate_hz, len_sec, n_filters):
    # Set parameters
    center_freq_hz_min = 100
    n_center_freqs = 24
    len_sample = int(np.floor(samplerate_hz * len_sec))

    # Initialize variables
    index = 0
    filterbank = np.zeros((n_filters, len_sample))
    current_center_freq_hz = center_freq_hz_min

    # Determine number of phase shifts per center frequency
    phase_pair_count = (
        np.ones(n_center_freqs) * np.floor(n_filters / 2 / n_center_freqs)
    ).astype(int)
    remaining_phase_pairs = (
                (n_filters - np.sum(phase_pair_count) * 2) / 2
    ).astype(int)
    if remaining_phase_pairs > 0:
        phase_pair_count[:remaining_phase_pairs] = (
            phase_pair_count[:remaining_phase_pairs] + 1
        )

    # Generate all filters for each center frequencies
    for i in range(n_center_freqs):
        # Generate all filters for all phase shifts
        for phase_index in range(phase_pair_count[i]):
            # First half of filtes: phase_shifts in [0,pi)
            current_phase_shift = (
                np.float(phase_index) / phase_pair_count[i] * np.pi
            )
            filterbank[index, :] = gammatone_impulse_response(
                samplerate_hz,
                len_sec,
                current_center_freq_hz,
                current_phase_shift,
            )
            index = index + 1

        # Second half of filters: phase_shifts in [pi, 2*pi)
        filterbank[index: index + phase_pair_count[i], :] = -filterbank[
            index - phase_pair_count[i]: index, :
        ]

        # Prepare for next center frequency
        index = index + phase_pair_count[i]
        current_center_freq_hz = erb_scale_2_freq_hz(
            freq_hz_2_erb_scale(current_center_freq_hz) + 1
        )

    filterbank = normalize_filters(filterbank)
    return filterbank


def gammatone_impulse_response(samplerate_hz, len_sec, center_freq_hz,
                               phase_shift):
    """ Generate single parametrized gammatone filter """
    p = 2  # filter order
    erb = 24.7 + 0.108 * center_freq_hz  # equivalent rectangular bandwidth
    divisor = (
        np.pi * np.math.factorial(2 * p - 2) * np.power(2, float(-(2 * p - 2)))
    ) / np.square(np.math.factorial(p - 1))
    b = erb / divisor  # bandwidth parameter
    a = 1.0  # amplitude. This is varied later by the normalization process.
    len_sample = int(np.floor(samplerate_hz * len_sec))
    t = np.linspace(1.0 / samplerate_hz, len_sec, len_sample)
    gammatone_ir = (
        a
        * np.power(t, p - 1)
        * np.exp(-2 * np.pi * b * t)
        * np.cos(2 * np.pi * center_freq_hz * t + phase_shift)
    )
    return gammatone_ir


def erb_scale_2_freq_hz(freq_erb):
    """ Convert frequency on ERB scale to frequency in Hertz """
    freq_hz = (np.exp(freq_erb / 9.265) - 1) * 24.7 * 9.265
    return freq_hz


def freq_hz_2_erb_scale(freq_hz):
    """ Convert frequency in Hertz to frequency on ERB scale """
    freq_erb = 9.265 * np.log(1 + freq_hz / (24.7 * 9.265))
    return freq_erb


def normalize_filters(filterbank):
    """ Normalizes a filterbank such that all filters
    have the same root mean square (RMS). """
    rms_per_filter = np.sqrt(np.mean(np.square(filterbank), axis=1))
    rms_normalization_values = 1.0 / (rms_per_filter / np.amax(rms_per_filter))
    normalized_filterbank = (
        filterbank * rms_normalization_values[:, np.newaxis]
    )
    return normalized_filterbank
