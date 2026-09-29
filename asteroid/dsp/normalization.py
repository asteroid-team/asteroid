import numpy as np


def normalize_estimates(est_np, mix_np):
    """Normalizes estimates according to the mixture maximum amplitude

    Args:
        est_np (np.array): Estimates with shape (n_src, time).
        mix_np (np.array): One mixture with shape (time, ).

    """
    mix_max = np.max(np.abs(mix_np))

    def _scale(est):
        peak = np.max(np.abs(est))
        # A silent source has a peak of 0. Dividing by that peak returned NaN.
        if peak == 0:
            return est
        return est * mix_max / peak

    return np.stack([_scale(est) for est in est_np])
