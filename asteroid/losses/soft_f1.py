import torch
from torch.nn.modules.loss import _Loss


class F1_loss(_Loss):
    """Calculate F1 score"""

    def __init__(self, eps=1e-10):
        super().__init__()
        self.eps = eps

    def forward(self, estimates, targets):
        tp = (targets * estimates).sum()
        fp = ((1 - targets) * estimates).sum()
        fn = (targets * (1 - estimates)).sum()

        precision = tp / (tp + fp + self.eps)
        recall = tp / (tp + fn + self.eps)

        f1 = 2 * (precision * recall) / (precision + recall + self.eps)
        return 1 - f1.mean()
