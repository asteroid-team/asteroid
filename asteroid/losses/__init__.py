from .pit_wrapper import PITLossWrapper
from .mixit_wrapper import MixITLossWrapper
from .sinkpit_wrapper import SinkPITLossWrapper
from .sdr import PairwiseNegSDR
from .sdr import pairwise_neg_sisdr, singlesrc_neg_sisdr, multisrc_neg_sisdr
from .sdr import pairwise_neg_sdsdr, singlesrc_neg_sdsdr, multisrc_neg_sdsdr
from .sdr import pairwise_neg_snr, singlesrc_neg_snr, multisrc_neg_snr
from .mse import pairwise_mse, singlesrc_mse, multisrc_mse
from .cluster import deep_clustering_loss
from .pmsqe import SingleSrcPMSQE
from .multi_scale_spectral import SingleSrcMultiScaleSpectral

try:
    from .stoi import NegSTOILoss as SingleSrcNegSTOI
except ModuleNotFoundError:
    # Is installed with asteroid, but remove the deps for TorchHub.
    def f():
        raise ModuleNotFoundError("No module named 'torch_stoi'")

    SingleSrcNegSTOI = lambda *a, **kw: f()


__all__ = [
    "PITLossWrapper",
    "MixITLossWrapper",
    "SinkPITLossWrapper",
    "PairwiseNegSDR",
    "singlesrc_neg_sisdr",
    "pairwise_neg_sisdr",
    "multisrc_neg_sisdr",
    "pairwise_neg_sdsdr",
    "singlesrc_neg_sdsdr",
    "multisrc_neg_sdsdr",
    "pairwise_neg_snr",
    "singlesrc_neg_snr",
    "multisrc_neg_snr",
    "pairwise_mse",
    "singlesrc_mse",
    "multisrc_mse",
    "deep_clustering_loss",
    "SingleSrcPMSQE",
    "SingleSrcNegSTOI",
    "SingleSrcMultiScaleSpectral",
]
