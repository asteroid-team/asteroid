import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import LSTM, Linear, BatchNorm1d, Parameter
from .base_models import BaseModel


class XUMX(BaseModel):
    r"""CrossNet-Open-Unmix (X-UMX) for Music Source Separation introduced in [1].
        There are two notable contributions with no effect on inference:
            a) Multi Domain Losses
                - Considering not only spectrograms but also time signals
            b) Combination Scheme
                - Considering possible combinations of output instruments
        When starting to train X-UMX, you can optionally use the above by setting
        ``loss_use_multidomain'' and ``loss_combine_sources'' which are both set in conf.yml.

    Args:
        sources (list): The list of instruments, e.g., ["bass", "drums", "vocals"],
            defined in conf.yml.
        window_length (int): The length in samples of window function to use in STFT.
        in_chan (int): Number of input channels, should be equal to
            STFT size and STFT window length in samples.
        n_hop (int): STFT hop length in samples.
        hidden_size (int): Hidden size parameter of LSTM layers.
        nb_channels (int): set number of channels for model (1 for mono
            (spectral downmix is applied,) 2 for stereo).
        sample_rate (int): sampling rate of input wavs
        nb_layers (int): Number of (B)LSTM layers in network.
        input_mean (torch.tensor): Mean for each frequency bin calculated
            in advance to normalize the mixture magnitude spectrogram.
        input_scale (torch.tensor): Standard deviation for each frequency bin
            calculated in advance to normalize the mixture magnitude spectrogram.
        max_bin (int): Maximum frequency bin index of the mixture that X-UMX
            should consider. Set to None to use all frequency bins.
        bidirectional (bool): whether we use LSTM or BLSTM.
        spec_power (int): Exponent for spectrogram calculation.
        return_time_signals (bool): Set to true if you are using a time-domain
            loss., i.e., applies ISTFT. If you select ``MDL=True'' via
            conf.yml, this is set as True.

    References
        [1] "All for One and One for All: Improving Music Separation by Bridging
        Networks", Ryosuke Sawata, Stefan Uhlich, Shusuke Takahashi and Yuki Mitsufuji.
        https://arxiv.org/abs/2010.04228 (and ICASSP 2021)
    """

    def __init__(
        self,
        sources,
        window_length=4096,
        in_chan=4096,
        n_hop=1024,
        hidden_size=512,
        nb_channels=2,
        sample_rate=44100,
        nb_layers=3,
        input_mean=None,
        input_scale=None,
        max_bin=None,
        bidirectional=True,
        spec_power=1,
        return_time_signals=False,
    ):
        super().__init__(sample_rate)

        self.window_length = window_length
        self.in_chan = in_chan
        self.n_hop = n_hop
        self.sources = sources
        self._return_time_signals = return_time_signals
        self.nb_channels = nb_channels
        self.nb_layers = nb_layers
        self.bidirectional = bidirectional
        self.nb_output_bins = in_chan // 2 + 1
        if max_bin:
            self.max_bin = max_bin
        else:
            self.max_bin = self.nb_output_bins
        self.hidden_size = hidden_size
        self.spec_power = spec_power

        if input_mean is not None:
            input_mean = torch.from_numpy(-input_mean[: self.max_bin]).float()
        else:
            input_mean = torch.zeros(self.max_bin)

        if input_scale is not None:
            input_scale = torch.from_numpy(1.0 / input_scale[: self.max_bin]).float()
        else:
            input_scale = torch.ones(self.max_bin)

        # Define spectral encoder
        stft = _STFT(window_length=window_length, n_fft=in_chan, n_hop=n_hop, center=True)
        spec = _Spectrogram(spec_power=spec_power, mono=(nb_channels == 1))
        self.encoder = nn.Sequential(stft, spec)  # Return: Spec, Angle

        # Define DNN Core
        lstm_hidden_size = hidden_size // 2 if bidirectional else hidden_size
        src_enc = {}
        src_lstm = {}
        src_dec = {}
        mean_scale = {}
        for src in sources:
            # Define Enc.
            src_enc[src] = _InstrumentBackboneEnc(
                nb_bins=self.max_bin,
                hidden_size=hidden_size,
                nb_channels=nb_channels,
            )

            # Define Recurrent Lyaers.
            src_lstm[src] = LSTM(
                input_size=hidden_size,
                hidden_size=lstm_hidden_size,
                num_layers=nb_layers,
                bidirectional=bidirectional,
                batch_first=False,
                dropout=0.4,
            )

            # Define Dec.
            src_dec[src] = _InstrumentBackboneDec(
                nb_output_bins=self.nb_output_bins,
                hidden_size=hidden_size,
                nb_channels=nb_channels,
            )

            mean_scale["input_mean_{}".format(src)] = Parameter(input_mean.clone())
            mean_scale["input_scale_{}".format(src)] = Parameter(input_scale.clone())
            mean_scale["output_mean_{}".format(src)] = Parameter(
                torch.ones(self.nb_output_bins).float()
            )
            mean_scale["output_scale_{}".format(src)] = Parameter(
                torch.ones(self.nb_output_bins).float()
            )
        self.layer_enc = nn.ModuleDict(src_enc)
        self.layer_lstm = nn.ModuleDict(src_lstm)
        self.layer_dec = nn.ModuleDict(src_dec)
        self.mean_scale = nn.ParameterDict(mean_scale)

        # Define spectral decoder
        self.decoder = _ISTFT(window=stft.window, n_fft=in_chan, hop_length=n_hop, center=True)

    def forward(self, wav):
        """Model forward

        Args:
            wav (torch.Tensor): waveform tensor. 1D, 2D or 3D tensor, time last.

        Returns:
            masked_mixture (torch.Tensor): estimated spectrograms masked by
                X-UMX's output of shape $(sources, frames, batch_size, channels, bins)$
            time_signals (torch.Tensor): estimated time signals of shape $(sources, batch_size, channels, time_length)$ if `return_time_signals` is `True`
        """
        # Transform
        mixture, ang = self.encoder(wav)

        # Estimate masks
        est_masks = self.forward_masker(mixture.clone())

        # Apply masks to mixture
        masked_mixture = self.apply_masks(mixture, est_masks)

        # Inverse Transform
        if self._return_time_signals:
            spec = masked_mixture.permute(0, 2, 3, 4, 1)
            time_signals = self.decoder(spec, ang)
        else:
            time_signals = None

        return masked_mixture, time_signals

    def forward_masker(self, input_spec):
        shapes = input_spec.data.shape

        # crop
        x = input_spec[..., : self.max_bin]

        # clone for the number of sources
        inputs = [x]
        for i in range(1, len(self.sources)):
            inputs.append(x.clone())

        # shift and scale input to mean=0 std=1 (across all bins)
        # and encode to (nb_frames*nb_samples, hidden_size)
        for i, src in enumerate(self.sources):
            inputs[i] += self.mean_scale["input_mean_{}".format(src)]
            inputs[i] *= self.mean_scale["input_scale_{}".format(src)]
            inputs[i] = self.layer_enc[src](inputs[i], shapes)

        # 1st Bridging operation and apply 3-layers of stacked LSTM
        cross_1 = sum(inputs) / len(self.sources)
        cross_2 = 0.0
        for i, src in enumerate(self.sources):
            tmp_lstm_out = self.layer_lstm[src](cross_1)
            # lstm skip connection
            cross_2 += torch.cat([inputs[i], tmp_lstm_out[0]], -1)

        # 2nd Bridging operation
        cross_2 /= len(self.sources)
        mask_list = []
        for src in self.sources:
            x_tmp = self.layer_dec[src](cross_2, shapes)
            x_tmp *= self.mean_scale["output_scale_{}".format(src)]
            x_tmp += self.mean_scale["output_mean_{}".format(src)]
            mask_list.append(F.relu(x_tmp))
        est_masks = torch.stack(mask_list, dim=0)

        return est_masks

    def apply_masks(self, mixture, est_masks):
        masked_tf_rep = torch.stack([mixture * est_masks[i] for i in range(len(self.sources))])
        return masked_tf_rep

    def get_model_args(self):
        """Arguments needed to re-instantiate the model."""
        fb_config = {
            "window_length": self.window_length,
            "in_chan": self.in_chan,
            "n_hop": self.n_hop,
            "sample_rate": self.sample_rate,
        }

        net_config = {
            "sources": self.sources,
            "hidden_size": self.hidden_size,
            "nb_channels": self.nb_channels,
            "input_mean": None,
            "input_scale": None,
            "max_bin": self.max_bin,
            "nb_layers": self.nb_layers,
            "bidirectional": self.bidirectional,
            "spec_power": self.spec_power,
            "return_time_signals": False,
        }

        # Merge all args under model_args.
        model_args = {
            **fb_config,
            **net_config,
        }
        return model_args


class _InstrumentBackboneEnc(nn.Module):
    """Encoder structure that maps the mixture magnitude spectrogram to
    smaller-sized features which are the input for the LSTM layers.

    Args:
        nb_bins (int): Number of frequency bins of the mixture.
        hidden_size (int): Hidden size parameter of LSTM layers.
        nb_channels (int): set number of channels for model
            (1 for mono (spectral downmix is applied,) 2 for stereo).
    """

    def __init__(
        self,
        nb_bins,
        hidden_size=512,
        nb_channels=2,
    ):
        super().__init__()

        self.max_bin = nb_bins
        self.hidden_size = hidden_size
        self.enc = nn.Sequential(
            Linear(self.max_bin * nb_channels, hidden_size, bias=False),
            BatchNorm1d(hidden_size),
        )

    def forward(self, x, shapes):
        nb_frames, nb_samples, nb_channels, _ = shapes
        x = self.enc(x.reshape(-1, nb_channels * self.max_bin))
        x = x.reshape(nb_frames, nb_samples, self.hidden_size)

        # squash range to [-1, 1]
        x = torch.tanh(x)
        return x


class _InstrumentBackboneDec(nn.Module):
    """Decoder structure that maps output of LSTM layers to
    magnitude estimate of an instrument.

    Args:
        nb_output_bins (int): Number of frequency bins of the instrument estimate.
        hidden_size (int): Hidden size parameter of LSTM layers.
        nb_channels (int): Number of output bins depending on STFT size.
            It is generally calculated ``(STFT size) // 2 + 1''.
    """

    def __init__(
        self,
        nb_output_bins,
        hidden_size=512,
        nb_channels=2,
    ):
        super().__init__()
        self.nb_output_bins = nb_output_bins
        self.dec = nn.Sequential(
            Linear(in_features=hidden_size * 2, out_features=hidden_size, bias=False),
            BatchNorm1d(hidden_size),
            nn.ReLU(),
            Linear(
                in_features=hidden_size, out_features=self.nb_output_bins * nb_channels, bias=False
            ),
            BatchNorm1d(self.nb_output_bins * nb_channels),
        )

    def forward(self, x, shapes):
        nb_frames, nb_samples, nb_channels, _ = shapes
        x = self.dec(x.reshape(-1, x.shape[-1]))
        x = x.reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins)
        return x


class _STFT(nn.Module):
    def __init__(self, window_length, n_fft=4096, n_hop=1024, center=True):
        super(_STFT, self).__init__()
        self.window = Parameter(torch.hann_window(window_length), requires_grad=False)
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.center = center

    def forward(self, x):
        """
        Input: (nb_samples, nb_channels, nb_timesteps)
        Output:(nb_samples, nb_channels, nb_bins, nb_frames, 2)
        """

        nb_samples, nb_channels, nb_timesteps = x.size()

        # merge nb_samples and nb_channels for multichannel stft
        x = x.reshape(nb_samples * nb_channels, -1)

        # compute stft with parameters as close as possible scipy settings
        stft_f = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.n_hop,
            window=self.window,
            center=self.center,
            normalized=False,
            onesided=True,
            pad_mode="reflect",
            return_complex=True,
        )
        stft_f = torch.view_as_real(stft_f)

        # reshape back to channel dimension
        stft_f = stft_f.contiguous().view(nb_samples, nb_channels, self.n_fft // 2 + 1, -1, 2)
        return stft_f


class _Spectrogram(nn.Module):
    def __init__(self, spec_power=1, mono=True):
        super(_Spectrogram, self).__init__()
        self.spec_power = spec_power
        self.mono = mono

    def forward(self, stft_f):
        """
        Input: complex STFT
            (nb_samples, nb_channels, nb_bins, nb_frames, 2)
        Output: Power/Mag Spectrogram and the corresponding phase
            (nb_frames, nb_samples, nb_channels, nb_bins)
        """

        phase = stft_f.detach().clone()
        phase = torch.atan2(phase[Ellipsis, 1], phase[Ellipsis, 0])

        stft_f = stft_f.transpose(2, 3)

        # take the magnitude
        stft_f = stft_f.pow(2).sum(-1).pow(self.spec_power / 2.0)

        # downmix in the mag domain
        if self.mono:
            stft_f = torch.mean(stft_f, 1, keepdim=True)
            phase = torch.mean(phase, 1, keepdim=True)

        # permute output for LSTM convenience
        return [stft_f.permute(2, 0, 1, 3), phase]


class _ISTFT(nn.Module):
    def __init__(self, window, n_fft=4096, hop_length=1024, center=True):
        super(_ISTFT, self).__init__()
        self.window = window
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.center = center

    def forward(self, spec, ang):
        sources, bsize, channels, fbins, frames = spec.shape
        x_r = spec * torch.cos(ang)
        x_i = spec * torch.sin(ang)
        x = torch.stack([x_r, x_i], dim=-1)
        x = x.view(sources * bsize * channels, fbins, frames, 2)
        x = torch.view_as_complex(x)
        wav = torch.istft(
            x, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, center=self.center
        )
        wav = wav.view(sources, bsize, channels, wav.shape[-1])

        return wav
