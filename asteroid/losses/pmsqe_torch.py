# Pytorch implementation of the Perceptual Metric for Speech Quality Evaluation (PMSQE). This metric is
# computed per frame from the magnitude spectra of the reference and processed speech signal.
# Inspired on the Perceptual Evaluation of the Speech Quality (PESQ) algorithm, this loss function consists of
# two regularization factors which account for the symmetrical and asymmetrical distortion in the loudness domain.
# See Tensorflow implementation and [1] for more details.
#
# Implemented by Juan M. Martin. Contact: mdjuamart@ugr.es
#
#   References:
#    [1] J.M.Martin, A.M.Gomez, J.A.Gonzalez, A.M.Peinado 'A Deep Learning Loss Function based on the Perceptual
#    Evaluation of the Speech Quality', IEEE Signal Processing Letters, 2018.
#
#
# Copyright 2019: University of Granada, Signal Processing, Multimedia Transmission and Speech/Audio Technologies
# (SigMAT) Group. The software is free for non-commercial use. This program comes WITHOUT ANY WARRANTY.
#


import numpy as np

import torch
from torch import Tensor
import torch.nn as nn

from scipy.io import loadmat


class PMSQE(nn.Module):
    """
    Pytorch class which implements the PMSQE loss function described in our paper [1].
    This version is only designed for 16 kHz (512 length DFT).
    Adaptation to 8 kHz could be done by changing the parameters of the class (see Tensorflow implementation).
    Also, the SLL, frequency and gain equalization are applied in each sequence independently (padding ignored).

    Parameters:
        window_length: Select the used window function for the correct factor to be applied.
        window_weight: (Optional) correction to the window factor applied.
        bark_eq: Boolean to indicate if apply bark equalization.
        gain_eq: Boolean to indicate if apply gain equalization.
    """


    def __init__(self, window_name = 'sqrt_hann', window_weight = 1.0, bark_eq = True, gain_eq = True):

        super(PMSQE, self).__init__()

        self.window_name = window_name
        self.window_weight = window_weight
        self.bark_eq = bark_eq
        self.gain_eq = gain_eq

        self.Sp = 6.910853e-006
        self.Sl = 1.866055e-001

        self.nbins = 512
        self.nbark = 49

        if self.window_name == 'rect':
            self.pow_correc_factor = 1.0
        elif self.window_name == 'hann':
            self.pow_correc_factor = 2.666666666666754
        elif self.window_name == 'sqrt_hann':
            self.pow_correc_factor = 2.0
        elif self.window_name == 'hamming':
            self.pow_correc_factor = 2.51635879188799
        elif self.window_name == 'flatTop':
            self.pow_correc_factor = 5.70713295690759
        else:
            raise ValueError('Unexpected window type')

        self.pow_correc_factor = self.pow_correc_factor * self.window_weight

        self.abs_thresh_power = nn.Parameter(Tensor([
            51286152.00, 2454709.500, 70794.593750,
            4897.788574, 1174.897705, 389.045166,
            104.712860, 45.708820, 17.782795,
            9.772372, 4.897789, 3.090296,
            1.905461, 1.258925, 0.977237,
            0.724436, 0.562341, 0.457088,
            0.389045, 0.331131, 0.295121,
            0.269153, 0.257040, 0.251189,
            0.251189, 0.251189, 0.251189,
            0.263027, 0.288403, 0.309030,
            0.338844, 0.371535, 0.398107,
            0.436516, 0.467735, 0.489779,
            0.501187, 0.501187, 0.512861,
            0.524807, 0.524807, 0.524807,
            0.512861, 0.478630, 0.426580,
            0.371535, 0.363078, 0.416869,
            0.537032]), requires_grad = False)

        self.modified_zwicker_power = nn.Parameter(Tensor([0.25520097857560436, 0.25520097857560436,
                              0.25520097857560436, 0.25520097857560436,
                              0.25168783742879913, 0.24806665731869609,
                              0.244767379124259, 0.24173800119368227,
                              0.23893798876066405, 0.23633516221479894,
                              0.23390360348392067, 0.23162209128929445,
                              0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23,
                              0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23,
                              0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23,
                              0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23,
                              0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23,
                              0.23, 0.23]), requires_grad = False)

        self.width_of_band_bark = nn.Parameter(Tensor([
            0.157344, 0.317994, 0.322441, 0.326934, 0.331474,
            0.336061, 0.340697, 0.345381, 0.350114, 0.354897,
            0.359729, 0.364611, 0.369544, 0.374529, 0.379565,
            0.384653, 0.389794, 0.394989, 0.400236, 0.405538,
            0.410894, 0.416306, 0.421773, 0.427297, 0.432877,
            0.438514, 0.444209, 0.449962, 0.455774, 0.461645,
            0.467577, 0.473569, 0.479621, 0.485736, 0.491912,
            0.498151, 0.504454, 0.510819, 0.517250, 0.523745,
            0.530308, 0.536934, 0.543629, 0.550390, 0.557220,
            0.564119, 0.571085, 0.578125, 0.585232]), requires_grad = False)

        self.sqrt_total_width = torch.sqrt(torch.sum(self.width_of_band_bark))

        self.bark_matrix = nn.Parameter(Tensor(loadmat('bark_matrix_16k.mat')["Bark_matrix_16k"]),
                                               requires_grad=False)

        mask_sll = np.zeros(shape=[self.nbins / 2 + 1], dtype=np.float32)
        mask_sll[11] = 0.5 * 25.0 / 31.25
        mask_sll[12:104] = 1.0
        mask_sll[104] = 0.5
        mask_sll = mask_sll * self.pow_correc_factor * (self.nbins+2.0)/(self.nbins*self.nbins)
        self.mask_sll = nn.Parameter(Tensor(mask_sll), requires_grad = False)


    def forward(self, output, target, pad_mask):
        """

        :param output: Dimensions (B,T,F). Padded degraded power spectrum in time-frequency domain.
        :param target: Dimensions (B,T,F). Zero-Padded reference power spectrum in time-frequency domain.
        :param pad_mask:  Dimensions (B,T,1). Mask to indicate the padding frames (1 indicates valid frame and
                          0 padding). Float tensor.

        Dimensions:
            B: Number of sequences in the batch.
            T: Number of time frames.
            F: Number of frequency bins.

        :return:
            wD: Mean symmetric distortion.
            wDA: Mean asymmetric distortion.
        """

        # SLL equalization
        ref_spectra = self.magnitude_at_standard_listening_level(target, pad_mask)
        deg_spectra = self.magnitude_at_standard_listening_level(output, pad_mask)

        # Bark spectra computation
        ref_bark_spectra = self.bark_computation(ref_spectra)
        deg_bark_spectra = self.bark_computation(deg_spectra)

        # (Optional) frequency and gain equalization
        if self.bark_eq:
            deg_bark_spectra = self.bark_frequency_equalization(ref_bark_spectra, deg_bark_spectra)

        if self.gain_eq:
            deg_bark_spectra = self.bark_gain_equalization(ref_bark_spectra, deg_bark_spectra)

        # Distortion matrix computation
        D, DA = self.compute_distortion_tensors(ref_bark_spectra, deg_bark_spectra)

        # Per-frame distortion
        audible_power_ref = self.compute_audible_power(ref_bark_spectra, 1.0)
        wD_frame, wDA_frame = self.per_frame_distortion(D, DA, audible_power_ref)

        # Mean distortions over frames (taking into account padding)
        wD = torch.sum (wD_frame * pad_mask) / pad_mask.sum()
        wDA = torch.sum (wDA_frame * pad_mask) / pad_mask.sum()

        return wD, wDA


    def magnitude_at_standard_listening_level(self, spectra, pad_mask):

        # Apply padding and SLL masking
        masked_spectra = spectra * pad_mask * self.mask_sll

        # Compute mean over frequency
        freq_mean_masked_spectra = torch.mean(masked_spectra, dim=-1, keepdim=True)

        # Compute mean over time (taking into account padding)
        sum_spectra = torch.sum(freq_mean_masked_spectra, dim=-2, keepdim = True)
        seq_len = torch.sum(pad_mask, dim=-2, keepdim = True)
        mean_pow = sum_spectra / seq_len

        # Compute final SLL spectra
        sll_spectra = spectra * (10000000.0 / mean_pow)

        return sll_spectra


    def bark_computation(self, spectra):

        bark_spectra = self.Sp * torch.matmul(spectra, self.bark_matrix)

        return bark_spectra


    def compute_audible_power(self, bark_spectra, factor = 1.0):

        # Apply absolute hearing threshold to each band
        thr_bark_spectra = torch.where(bark_spectra > self.abs_thresh_power * factor,
                                       bark_spectra, torch.zeros_like(bark_spectra))

        # Sum band power over frequency
        audible_power = torch.sum(thr_bark_spectra, dim=-1, keepdim=True)

        return audible_power


    def bark_gain_equalization(self, ref_bark_spectra, deg_bark_spectra):

        # Compute audible power
        audible_power_ref = self.compute_audible_power(ref_bark_spectra, 1.0)
        audible_power_deg = self.compute_audible_power(deg_bark_spectra, 1.0)

        # Compute gain factor
        gain = (audible_power_ref + 5.0e3) / (audible_power_deg + 5.0e3)

        # Limit the range of the gain factor
        limited_gain = torch.min(gain, 5.0 * torch.ones_like(gain))
        limited_gain = torch.max(limited_gain, 3.0e-4 * torch.ones_like(limited_gain))

        # Apply gain correction on degraded
        eq_deg_bark_spectra = limited_gain * deg_bark_spectra

        return eq_deg_bark_spectra


    def bark_frequency_equalization(self, ref_bark_spectra, deg_bark_spectra):

        # This version is applied in the degraded directly

        # Identification of speech active frames
        audible_powerX100 = self.compute_audible_power(ref_bark_spectra, 100.0)
        not_silent = audible_powerX100 >= 1.0e7

        # Threshold for active bark bins
        cond_thr = ref_bark_spectra >= self.abs_thresh_power * 100.0
        ref_thresholded = torch.where(cond_thr, ref_bark_spectra, torch.zeros_like(ref_bark_spectra))
        deg_thresholded = torch.where(cond_thr, deg_bark_spectra, torch.zeros_like(deg_bark_spectra))

        # Total power per bark bin
        avg_pow_per_bark_ref = torch.sum(
            torch.where(not_silent, ref_thresholded, torch.zeros_like(ref_thresholded)),
            dim=-2, keepdim=True)
        avg_pow_per_bark_deg = torch.sum(
            torch.where(not_silent, deg_thresholded, torch.zeros_like(deg_thresholded)),
            dim=-2, keepdim=True)

        # Compute equalizer
        equalizer = (avg_pow_per_bark_ref + 1000.0) / (avg_pow_per_bark_deg + 1000.0)
        equalizer = torch.min(equalizer, 100.0 * torch.ones_like(equalizer))
        equalizer = torch.max(equalizer, 0.01 * torch.ones_like(equalizer))

        # Apply frequency correction on degraded
        eq_deg_bark_spectra = equalizer * deg_bark_spectra

        return eq_deg_bark_spectra


    def loudness_computation(self, bark_spectra):

        # Bark spectra are transformed to a sone loudness scale using Zwicker's law
        loudness_dens = self.Sl * torch.pow(self.abs_thresh_power / 0.5, self.modified_zwicker_power) * (torch.pow(
            0.5 + 0.5 * bark_spectra / self.abs_thresh_power, self.modified_zwicker_power) - 1.0)

        cond = bark_spectra < self.abs_thresh_power
        loudness_dens_limited = torch.where(cond, torch.zeros_like(loudness_dens), loudness_dens)

        return loudness_dens_limited


    def compute_distortion_tensors(self, ref_bark_spectra, deg_bark_spectra):

        # After bark spectra are compensated, these are transformed to sone loudness
        original_loudness = self.loudness_computation(ref_bark_spectra)
        distorted_loudness = self.loudness_computation(deg_bark_spectra)

        # Loudness difference
        r = torch.abs(distorted_loudness - original_loudness)
        # Masking effect computation
        m = 0.25 * torch.min(original_loudness, distorted_loudness)
        # Center clipping using masking effect
        D = torch.max(r - m, torch.zeros_like(r))

        # Asymmetry factor computation
        Asym = torch.pow((deg_bark_spectra + 50.0) / (ref_bark_spectra + 50.0), 1.2)
        cond = Asym < 3.0 * torch.ones_like(Asym)
        AF = torch.where(cond, torch.zeros_like(Asym), torch.min(Asym, 12.0 * torch.ones_like(Asym)))
        # Asymmetric Disturbance matrix computation
        DA = AF * D

        return D, DA


    def per_frame_distortion(self, D, DA, total_power_ref):

        # Computation of the norms over bark bands for each frame (2 and 1 for D and DA, respectively)
        D_frame = torch.sqrt(torch.sum(torch.pow(D * self.width_of_band_bark, 2.0),
                                       dim=-1, keepdim=True)) * self.sqrt_total_width
        DA_frame = torch.sum(DA * self.width_of_band_bark, dim = -1, keepdim=True)

        # Weighting by the audible power raised to 0.04
        weights = torch.pow((total_power_ref + 1e5) / 1e7, 0.04)

        # Bounded computation of the per frame distortion metric
        wD_frame = torch.min(D_frame / weights, 45.0 * torch.ones_like(D_frame))
        wDA_frame = torch.min(DA_frame / weights, 45.0 * torch.ones_like(DA_frame))

        return wD_frame, wDA_frame