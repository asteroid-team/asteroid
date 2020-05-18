import torch
from torch import nn
from asteroid.losses import PITLossWrapper, PairwiseNegSDR, pairwise_mse


""" translated from FUSS """

class LogMse(nn.Module):
    def __init__(self, max_snr=1e6, bias_ref_signal=None):
        super(LogMse, self).__init__()
        self.max_snr = max_snr
        self.bias_ref_signal = bias_ref_signal

    def forward(self, separated, source):

        err_pow = torch.sum((source - separated)**2, dim=-1)
        snrfactor = 10. ** (-self.max_snr / 10.)
        if self.bias_ref_signal is None:
            ref_pow = torch.sum((source)**2, dim=-1)
        else:
            ref_pow = torch.sum((self.bias_ref_signal)**2, dim=-1)
        bias = snrfactor * ref_pow

        return 10. * torch.log10(bias + err_pow)


def log_mse(source, separated, max_snr=1e6, bias_ref_signal=None):
    """ redefined cos i am lazy """

    err_pow = torch.sum((source - separated) ** 2, dim=-1)
    snrfactor = 10. ** (-max_snr / 10.)
    if bias_ref_signal is None:
        ref_pow = torch.sum((source) ** 2, dim=-1)
    else:
        ref_pow = torch.sum((bias_ref_signal) ** 2, dim=-1)
    bias = snrfactor * ref_pow
    return 10. * torch.log10(bias + err_pow + 1e-8)


class CustomLoss(nn.Module):

    def __init__(self, active_clip=-30, inactive_clip=-20):
        super().__init__()
        self.active_clip = active_clip
        self.inactive_clip = inactive_clip
        self.sisdr = PITLossWrapper(LogMse(), mode="pw_pt")
        self.mse = log_mse

    def forward(self, est_targets, targets, mix=None):

        _, reordered = self.sisdr(est_targets, targets, return_est=True)

        with torch.no_grad():
            silent = (torch.norm(targets, dim=-1, p=2) < 1e-8)

        B, n_sources, S = reordered.size()


        if (~silent).all():

            active = self.mse(targets[~silent], reordered[~silent], max_snr=30)
            return active.sum() / (B*n_sources)

        elif (silent).all():

            mix_zero = mix.unsqueeze(1).repeat(1, n_sources, 1)[silent]
            silent = self.mse(targets[silent], reordered[silent],  max_snr=20,  bias_ref_signal=mix_zero)
            return silent.sum() / (B*n_sources)
        else:

            active = self.mse( targets[~silent], reordered[~silent], max_snr=30)
            mix_zero = mix.unsqueeze(1).repeat(1, n_sources, 1)[silent]
            silent = self.mse(targets[silent], reordered[silent],  max_snr=20, bias_ref_signal=mix_zero)
            return (silent.sum() + active.sum()) / (B*n_sources)