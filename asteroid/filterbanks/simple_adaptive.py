import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveEncoder1D(nn.Module):
    """
        A 1D convolutional block that transforms signal in wave form into higher
        dimension.

        Args:
            input shape: [batch, 1, n_samples]
            output shape: [batch, freq_res, n_samples//sample_res]
            freq_res: number of output frequencies for the encoding convolution
            sample_res: int, length of the encoding filter
    """

    def __init__(self, freq_res, sample_res):
        super().__init__()
        self.conv = nn.Conv1d(1,
                              freq_res,
                              sample_res,
                              stride=sample_res // 2,
                              padding=sample_res // 2)

    def forward(self, s):
        return F.relu(self.conv(s))


class AdaptiveDecoder1D(nn.Module):
    """ A 1D deconvolutional block that transforms encoded representation
    into wave form.
    input shape: [batch, freq_res, sample_res]
    output shape: [batch, 1, sample_res*n_samples]
    freq_res: number of output frequencies for the encoding convolution
    sample_res: length of the encoding filter
    """

    def __init__(self, freq_res, sample_res, n_sources):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(n_sources * freq_res,
                                         n_sources,
                                         sample_res,
                                         padding=sample_res // 2,
                                         stride=sample_res // 2,
                                         groups=n_sources,
                                         output_padding=(sample_res // 2) - 1)

    def forward(self, x):
        return self.deconv(x)


class AdaptiveEncoderDecoder(nn.Module):
    """
        Adaptive basis encoder and decoder with inference of ideal masks.
        Copied from: https://github.com/etzinis/two_step_mask_learning/

        Args:
            freq_res: The number of frequency like representations
            sample_res: The number of samples in kernel 1D convolutions
            n_sources: The number of sources
        References:
            Tzinis, E., Venkataramani, S., Wang, Z., Subakan, Y. C., and
            Smaragdis, P., "Two-Step Sound Source Separation:
            Training on Learned Latent Targets." In Acoustics, Speech
            and Signal Processing (ICASSP), 2020 IEEE International Conference.
            https://arxiv.org/abs/1910.09804
    """

    def __init__(self,
                 freq_res=256,
                 sample_res=21,
                 n_sources=2):
        super().__init__()
        self.freq_res = freq_res
        self.sample_res = sample_res
        self.mix_encoder = AdaptiveEncoder1D(freq_res, sample_res)
        self.decoder = AdaptiveDecoder1D(freq_res, sample_res, n_sources)
        self.n_sources = n_sources

    def get_target_masks(self, clean_sources):
        """
        Get target masks for the given clean sources
        :param clean_sources: [batch, n_sources, time_samples]
        :return: Ideal masks for the given sources:
        [batch, n_sources, time_samples//(sample_res // 2)]
        """
        enc_mask_list = [self.mix_encoder(clean_sources[:, i, :].unsqueeze(1))
                         for i in range(self.n_sources)]
        total_mask = torch.stack(enc_mask_list, dim=1)
        return F.softmax(total_mask, dim=1)

    def reconstruct(self, mixture):
        enc_mixture = self.mix_encoder(mixture.unsqueeze(1))
        return self.decoder(enc_mixture)

    def get_encoded_sources(self, mixture, clean_sources):
        enc_mixture = self.mix_encoder(mixture.unsqueeze(1))
        enc_masks = self.get_target_masks(clean_sources)
        s_recon_enc = enc_masks * enc_mixture.unsqueeze(1)
        return s_recon_enc

    def forward(self, mixture, clean_sources):
        enc_mixture = self.mix_encoder(mixture.unsqueeze(1))
        enc_masks = self.get_target_masks(clean_sources)

        s_recon_enc = enc_masks * enc_mixture.unsqueeze(1)
        recon_sources = self.decoder(s_recon_enc.view(s_recon_enc.shape[0],
                                                      -1,
                                                      s_recon_enc.shape[-1]))
        return recon_sources, enc_masks
