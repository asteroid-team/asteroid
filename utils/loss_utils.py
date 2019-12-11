import torch


class DiscriminativeLoss(torch.nn.Module):

    def __init__(self, gamma=0.1):
        super(DiscriminativeLoss, self).__init__()

        self.gamma = gamma
        self.mse_loss = torch.nn.MSELoss(reduction="mean")

    def forward(self, input, target):
        swapped_target = target.clone().detach()
        swapped_target[..., 0], swapped_target[..., 1] = target[..., 1], target[..., 0]

        loss = self.mse_loss(input, target)
        loss -= self.gamma * self.mse_loss(input, swapped_target)

        return loss
