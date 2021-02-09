import numpy as np


def normalize_estimates(est_np, mix_np):
    """Normalizes estimates according to the mixture maximum amplitude

    Args:
        est_np (np.array): Estimates with shape (n_src, time).
        mix_np (np.array): One mixture with shape (time, ).

    """
    mix_max = np.max(np.abs(mix_np))
    return np.stack([est * mix_max / np.max(np.abs(est)) for est in est_np])
