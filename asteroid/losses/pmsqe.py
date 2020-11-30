import numpy as np
import torch
from torch import tensor
import torch.nn as nn
import pathlib
import os


class SingleSrcPMSQE(nn.Module):
    """Computes the Perceptual Metric for Speech Quality Evaluation (PMSQE)
    as described in [1].
    This version is only designed for 16 kHz (512 length DFT).
    Adaptation to 8 kHz could be done by changing the parameters of the
    class (see Tensorflow implementation).
    The SLL, frequency and gain equalization are applied in each
    sequence independently.

    Parameters:
        window_name (str): Select the used window function for the correct
            factor to be applied. Defaults to sqrt hanning window.
            Among ['rect', 'hann', 'sqrt_hann', 'hamming', 'flatTop'].
        window_weight (float, optional): Correction to the window factor
            applied.
        bark_eq (bool, optional): Whether to apply bark equalization.
        gain_eq (bool, optional): Whether to apply gain equalization.
        sample_rate (int): Sample rate of the input audio.

    References
        [1] J.M.Martin, A.M.Gomez, J.A.Gonzalez, A.M.Peinado 'A Deep Learning
        Loss Function based on the Perceptual Evaluation of the
        Speech Quality', IEEE Signal Processing Letters, 2018.
        Implemented by Juan M. Martin. Contact: mdjuamart@ugr.es

        Copyright 2019: University of Granada, Signal Processing, Multimedia
        Transmission and Speech/Audio Technologies (SigMAT) Group.

    .. note:: Inspired on the Perceptual Evaluation of the Speech Quality (PESQ)
        algorithm, this function consists of two regularization factors :
        the symmetrical and asymmetrical distortion in the loudness domain.

    Examples
        >>> import torch
        >>> from asteroid_filterbanks import STFTFB, Encoder, transforms
        >>> from asteroid.losses import PITLossWrapper, SingleSrcPMSQE
        >>> stft = Encoder(STFTFB(kernel_size=512, n_filters=512, stride=256))
        >>> # Usage by itself
        >>> ref, est = torch.randn(2, 1, 16000), torch.randn(2, 1, 16000)
        >>> ref_spec = transforms.mag(stft(ref))
        >>> est_spec = transforms.mag(stft(est))
        >>> loss_func = SingleSrcPMSQE()
        >>> loss_value = loss_func(est_spec, ref_spec)
        >>> # Usage with PITLossWrapper
        >>> loss_func = PITLossWrapper(SingleSrcPMSQE(), pit_from='pw_pt')
        >>> ref, est = torch.randn(2, 3, 16000), torch.randn(2, 3, 16000)
        >>> ref_spec = transforms.mag(stft(ref))
        >>> est_spec = transforms.mag(stft(est))
        >>> loss_value = loss_func(ref_spec, est_spec)
    """

    def __init__(
        self,
        window_name="sqrt_hann",
        window_weight=1.0,
        bark_eq=True,
        gain_eq=True,
        sample_rate=16000,
    ):
        super().__init__()
        self.window_name = window_name
        self.window_weight = window_weight
        self.bark_eq = bark_eq
        self.gain_eq = gain_eq

        if sample_rate not in [16000, 8000]:
            raise ValueError("Unsupported sample rate {}".format(sample_rate))
        self.sample_rate = sample_rate
        if sample_rate == 16000:
            self.Sp = 6.910853e-006
            self.Sl = 1.866055e-001
            self.nbins = 512
            self.nbark = 49
        else:
            self.Sp = 2.764344e-5
            self.Sl = 1.866055e-1
            self.nbins = 256
            self.nbark = 42
        # As described in [1] and used in the TF implementation.
        self.alpha = 0.1
        self.beta = 0.309 * self.alpha

        pow_correc_factor = self.get_correction_factor(window_name)
        self.pow_correc_factor = pow_correc_factor * self.window_weight
        # Initialize to None and populate as a function of sample rate.
        self.abs_thresh_power = None
        self.modified_zwicker_power = None
        self.width_of_band_bark = None
        self.bark_matrix = None
        self.mask_sll = None
        self.populate_constants(self.sample_rate)
        self.sqrt_total_width = torch.sqrt(torch.sum(self.width_of_band_bark))
        self.EPS = 1e-8

    def forward(self, est_targets, targets, pad_mask=None):
        """
        Args
            est_targets (torch.Tensor): Dimensions (B, T, F).
                Padded degraded power spectrum in time-frequency domain.
            targets (torch.Tensor): Dimensions (B, T, F).
                Zero-Padded reference power spectrum in time-frequency domain.
            pad_mask (torch.Tensor, optional):  Dimensions (B, T, 1). Mask
                to indicate the padding frames. Defaults to all ones.

        Dimensions
            B: Number of sequences in the batch.
            T: Number of time frames.
            F: Number of frequency bins.

        Returns
            torch.tensor of shape (B, ), wD + 0.309 * wDA

        ..note:: Dimensions (B, F, T) are also supported by SingleSrcPMSQE but are
            less efficient because input tensors are transposed (not inplace).

        """
        assert est_targets.shape == targets.shape
        # Need transpose? Find it out
        try:
            freq_idx = est_targets.shape.index(self.nbins // 2 + 1)
        except ValueError:
            raise ValueError(
                "Could not find dimension with {} elements in "
                "input tensors, verify your inputs"
                "".format(self.nbins // 2 + 1)
            )
        if freq_idx == 1:
            est_targets = est_targets.transpose(1, 2)
            targets = targets.transpose(1, 2)
        if pad_mask is not None:
            # Transpose the pad mask as well if needed.
            pad_mask = pad_mask.transpose(1, 2) if freq_idx == 1 else pad_mask
        else:
            # Suppose no padding if no pad_mask is provided.
            pad_mask = torch.ones(
                est_targets.shape[0], est_targets.shape[1], 1, device=est_targets.device
            )
        # SLL equalization
        ref_spectra = self.magnitude_at_sll(targets, pad_mask)
        deg_spectra = self.magnitude_at_sll(est_targets, pad_mask)

        # Bark spectra computation
        ref_bark_spectra = self.bark_computation(ref_spectra)
        deg_bark_spectra = self.bark_computation(deg_spectra)

        # (Optional) frequency and gain equalization
        if self.bark_eq:
            deg_bark_spectra = self.bark_freq_equalization(ref_bark_spectra, deg_bark_spectra)

        if self.gain_eq:
            deg_bark_spectra = self.bark_gain_equalization(ref_bark_spectra, deg_bark_spectra)

        # Distortion matrix computation
        sym_d, asym_d = self.compute_distortion_tensors(ref_bark_spectra, deg_bark_spectra)

        # Per-frame distortion
        audible_power_ref = self.compute_audible_power(ref_bark_spectra, 1.0)
        wd_frame, wda_frame = self.per_frame_distortion(sym_d, asym_d, audible_power_ref)
        # Mean distortions over frames : keep batch dims
        dims = [-1, -2]
        pmsqe_frame = (self.alpha * wd_frame + self.beta * wda_frame) * pad_mask
        pmsqe = torch.sum(pmsqe_frame, dim=dims) / pad_mask.sum(dims)
        return pmsqe

    def magnitude_at_sll(self, spectra, pad_mask):
        # Apply padding and SLL masking
        masked_spectra = spectra * pad_mask * self.mask_sll
        # Compute mean over frequency
        freq_mean_masked_spectra = torch.mean(masked_spectra, dim=-1, keepdim=True)
        # Compute mean over time (taking into account padding)
        sum_spectra = torch.sum(freq_mean_masked_spectra, dim=-2, keepdim=True)
        seq_len = torch.sum(pad_mask, dim=-2, keepdim=True)
        mean_pow = sum_spectra / seq_len
        # Compute final SLL spectra
        return 10000000.0 * spectra / mean_pow

    def bark_computation(self, spectra):
        return self.Sp * torch.matmul(spectra, self.bark_matrix)

    def compute_audible_power(self, bark_spectra, factor=1.0):
        # Apply absolute hearing threshold to each band
        thr_bark = torch.where(
            bark_spectra > self.abs_thresh_power * factor,
            bark_spectra,
            torch.zeros_like(bark_spectra),
        )
        # Sum band power over frequency
        return torch.sum(thr_bark, dim=-1, keepdim=True)

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
        return limited_gain * deg_bark_spectra

    def bark_freq_equalization(self, ref_bark_spectra, deg_bark_spectra):
        """This version is applied in the degraded directly."""
        # Identification of speech active frames
        audible_power_x100 = self.compute_audible_power(ref_bark_spectra, 100.0)
        not_silent = audible_power_x100 >= 1.0e7
        # Threshold for active bark bins
        cond_thr = ref_bark_spectra >= self.abs_thresh_power * 100.0
        ref_thresholded = torch.where(
            cond_thr, ref_bark_spectra, torch.zeros_like(ref_bark_spectra)
        )
        deg_thresholded = torch.where(
            cond_thr, deg_bark_spectra, torch.zeros_like(deg_bark_spectra)
        )
        # Total power per bark bin (ppb)
        avg_ppb_ref = torch.sum(
            torch.where(not_silent, ref_thresholded, torch.zeros_like(ref_thresholded)),
            dim=-2,
            keepdim=True,
        )
        avg_ppb_deg = torch.sum(
            torch.where(not_silent, deg_thresholded, torch.zeros_like(deg_thresholded)),
            dim=-2,
            keepdim=True,
        )
        # Compute equalizer
        equalizer = (avg_ppb_ref + 1000.0) / (avg_ppb_deg + 1000.0)
        equalizer = torch.min(equalizer, 100.0 * torch.ones_like(equalizer))
        equalizer = torch.max(equalizer, 0.01 * torch.ones_like(equalizer))
        # Apply frequency correction on degraded
        return equalizer * deg_bark_spectra

    def loudness_computation(self, bark_spectra):
        # Bark spectra transformed to a sone loudness scale using Zwicker's law
        aterm = torch.pow(self.abs_thresh_power / 0.5, self.modified_zwicker_power)
        bterm = (
            torch.pow(0.5 + 0.5 * bark_spectra / self.abs_thresh_power, self.modified_zwicker_power)
            - 1.0
        )
        loudness_dens = self.Sl * aterm * bterm
        cond = bark_spectra < self.abs_thresh_power
        return torch.where(cond, torch.zeros_like(loudness_dens), loudness_dens)

    def compute_distortion_tensors(self, ref_bark_spec, deg_bark_spec):
        # After bark spectra are compensated, transform to sone loudness
        original_loudness = self.loudness_computation(ref_bark_spec)
        distorted_loudness = self.loudness_computation(deg_bark_spec)
        # Loudness difference
        r = torch.abs(distorted_loudness - original_loudness)
        # Masking effect computation
        m = 0.25 * torch.min(original_loudness, distorted_loudness)
        # Center clipping using masking effect
        sym_d = torch.max(r - m, torch.ones_like(r) * self.EPS)
        # Asymmetry factor computation
        asym = torch.pow((deg_bark_spec + 50.0) / (ref_bark_spec + 50.0), 1.2)
        cond = asym < 3.0 * torch.ones_like(asym)
        asym_factor = torch.where(
            cond, torch.zeros_like(asym), torch.min(asym, 12.0 * torch.ones_like(asym))
        )
        # Asymmetric Disturbance matrix computation
        asym_d = asym_factor * sym_d
        return sym_d, asym_d

    def per_frame_distortion(self, sym_d, asym_d, total_power_ref):
        # Computation of the norms over bark bands for each frame
        # 2 and 1 for sym_d and asym_d, respectively
        d_frame = torch.sum(
            torch.pow(sym_d * self.width_of_band_bark, 2.0) + self.EPS, dim=-1, keepdim=True
        )
        # a = torch.pow(sym_d * self.width_of_band_bark, 2.0)
        # b = sym_d
        # print(a.min(),a.max(),b.min(),b.max(), d_frame.min(), d_frame.max())
        # print(self.width_of_band_bark.requires_grad)
        # print(d_frame.requires_grad)
        d_frame = torch.sqrt(d_frame) * self.sqrt_total_width
        da_frame = torch.sum(asym_d * self.width_of_band_bark, dim=-1, keepdim=True)
        # Weighting by the audible power raised to 0.04
        weights = torch.pow((total_power_ref + 1e5) / 1e7, 0.04)
        # Bounded computation of the per frame distortion metric
        wd_frame = torch.min(d_frame / weights, 45.0 * torch.ones_like(d_frame))
        wda_frame = torch.min(da_frame / weights, 45.0 * torch.ones_like(da_frame))
        return wd_frame, wda_frame

    @staticmethod
    def get_correction_factor(window_name):
        """ Returns the power correction factor depending on the window. """
        if window_name == "rect":
            return 1.0
        elif window_name == "hann":
            return 2.666666666666754
        elif window_name == "sqrt_hann":
            return 2.0
        elif window_name == "hamming":
            return 2.51635879188799
        elif window_name == "flatTop":
            return 5.70713295690759
        else:
            raise ValueError("Unexpected window type {}".format(window_name))

    def populate_constants(self, sample_rate):
        if sample_rate == 8000:
            self.register_8k_constants()
        elif sample_rate == 16000:
            self.register_16k_constants()
        # Mask SSL
        mask_sll = np.zeros(shape=[self.nbins // 2 + 1], dtype=np.float32)
        mask_sll[11] = 0.5 * 25.0 / 31.25
        mask_sll[12:104] = 1.0
        mask_sll[104] = 0.5
        correction = self.pow_correc_factor * (self.nbins + 2.0) / self.nbins ** 2
        mask_sll = mask_sll * correction
        self.mask_sll = nn.Parameter(tensor(mask_sll), requires_grad=False)

    def register_16k_constants(self):
        # Absolute threshold power
        abs_thresh_power = [
            51286152.00,
            2454709.500,
            70794.593750,
            4897.788574,
            1174.897705,
            389.045166,
            104.712860,
            45.708820,
            17.782795,
            9.772372,
            4.897789,
            3.090296,
            1.905461,
            1.258925,
            0.977237,
            0.724436,
            0.562341,
            0.457088,
            0.389045,
            0.331131,
            0.295121,
            0.269153,
            0.257040,
            0.251189,
            0.251189,
            0.251189,
            0.251189,
            0.263027,
            0.288403,
            0.309030,
            0.338844,
            0.371535,
            0.398107,
            0.436516,
            0.467735,
            0.489779,
            0.501187,
            0.501187,
            0.512861,
            0.524807,
            0.524807,
            0.524807,
            0.512861,
            0.478630,
            0.426580,
            0.371535,
            0.363078,
            0.416869,
            0.537032,
        ]
        self.abs_thresh_power = nn.Parameter(tensor(abs_thresh_power), requires_grad=False)
        # Modified zwicker power
        modif_zwicker_power = [
            0.25520097857560436,
            0.25520097857560436,
            0.25520097857560436,
            0.25520097857560436,
            0.25168783742879913,
            0.24806665731869609,
            0.244767379124259,
            0.24173800119368227,
            0.23893798876066405,
            0.23633516221479894,
            0.23390360348392067,
            0.23162209128929445,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
        ]
        self.modified_zwicker_power = nn.Parameter(tensor(modif_zwicker_power), requires_grad=False)
        # Width of band bark
        width_of_band_bark = [
            0.157344,
            0.317994,
            0.322441,
            0.326934,
            0.331474,
            0.336061,
            0.340697,
            0.345381,
            0.350114,
            0.354897,
            0.359729,
            0.364611,
            0.369544,
            0.374529,
            0.379565,
            0.384653,
            0.389794,
            0.394989,
            0.400236,
            0.405538,
            0.410894,
            0.416306,
            0.421773,
            0.427297,
            0.432877,
            0.438514,
            0.444209,
            0.449962,
            0.455774,
            0.461645,
            0.467577,
            0.473569,
            0.479621,
            0.485736,
            0.491912,
            0.498151,
            0.504454,
            0.510819,
            0.517250,
            0.523745,
            0.530308,
            0.536934,
            0.543629,
            0.550390,
            0.557220,
            0.564119,
            0.571085,
            0.578125,
            0.585232,
        ]
        self.width_of_band_bark = nn.Parameter(tensor(width_of_band_bark), requires_grad=False)
        # Bark matrix
        local_path = pathlib.Path(__file__).parent.absolute()
        bark_path = os.path.join(local_path, "bark_matrix_16k.mat")
        bark_matrix = self.load_mat(bark_path)["Bark_matrix_16k"].astype("float32")
        self.bark_matrix = nn.Parameter(tensor(bark_matrix), requires_grad=False)

    def register_8k_constants(self):
        # Absolute threshold power
        abs_thresh_power = [
            51286152,
            2454709.500,
            70794.593750,
            4897.788574,
            1174.897705,
            389.045166,
            104.712860,
            45.708820,
            17.782795,
            9.772372,
            4.897789,
            3.090296,
            1.905461,
            1.258925,
            0.977237,
            0.724436,
            0.562341,
            0.457088,
            0.389045,
            0.331131,
            0.295121,
            0.269153,
            0.257040,
            0.251189,
            0.251189,
            0.251189,
            0.251189,
            0.263027,
            0.288403,
            0.309030,
            0.338844,
            0.371535,
            0.398107,
            0.436516,
            0.467735,
            0.489779,
            0.501187,
            0.501187,
            0.512861,
            0.524807,
            0.524807,
            0.524807,
        ]
        self.abs_thresh_power = nn.Parameter(tensor(abs_thresh_power), requires_grad=False)
        # Modified zwicker power
        modif_zwicker_power = [
            0.25520097857560436,
            0.25520097857560436,
            0.25520097857560436,
            0.25520097857560436,
            0.25168783742879913,
            0.24806665731869609,
            0.244767379124259,
            0.24173800119368227,
            0.23893798876066405,
            0.23633516221479894,
            0.23390360348392067,
            0.23162209128929445,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
            0.23,
        ]
        self.modified_zwicker_power = nn.Parameter(tensor(modif_zwicker_power), requires_grad=False)
        # Width of band bark
        width_of_band_bark = [
            0.157344,
            0.317994,
            0.322441,
            0.326934,
            0.331474,
            0.336061,
            0.340697,
            0.345381,
            0.350114,
            0.354897,
            0.359729,
            0.364611,
            0.369544,
            0.374529,
            0.379565,
            0.384653,
            0.389794,
            0.394989,
            0.400236,
            0.405538,
            0.410894,
            0.416306,
            0.421773,
            0.427297,
            0.432877,
            0.438514,
            0.444209,
            0.449962,
            0.455774,
            0.461645,
            0.467577,
            0.473569,
            0.479621,
            0.485736,
            0.491912,
            0.498151,
            0.504454,
            0.510819,
            0.517250,
            0.523745,
            0.530308,
            0.536934,
        ]
        self.width_of_band_bark = nn.Parameter(tensor(width_of_band_bark), requires_grad=False)
        # Bark matrix
        local_path = pathlib.Path(__file__).parent.absolute()
        bark_path = os.path.join(local_path, "bark_matrix_8k.mat")
        bark_matrix = self.load_mat(bark_path)["Bark_matrix_8k"].astype("float32")
        self.bark_matrix = nn.Parameter(tensor(bark_matrix), requires_grad=False)

    def load_mat(self, *args, **kwargs):
        from scipy.io import loadmat

        return loadmat(*args, **kwargs)
