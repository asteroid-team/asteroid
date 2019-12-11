import torch


def snr(y_pred, y):
    y_inter = y_pred - y

    true_power = (y ** 2).sum()
    inter_power = (y_inter ** 2).sum()

    snr = 10*torch.log10(true_power / inter_power)

    return snr
