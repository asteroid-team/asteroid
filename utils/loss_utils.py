import torch


class DiscriminativeLoss(torch.nn.Module):

    def __init__(self, input_audio_size=2, gamma=0.1):
        super(DiscriminativeLoss, self).__init__()

        self.input_audio_size = input_audio_size
        self.gamma = gamma

    def forward(self, input, target):

        sum_mtr = torch.zeros_like(input[..., 0])
        for i in range(self.input_audio_size):
            sum_mtr += ((target[:,:,:,:,i]-input[:,:,:,:,i]) ** 2)
            for j in range(self.input_audio_size):
                if i != j:
                    sum_mtr -= (self.gamma * ((target[:,:,:,:,i]-input[:,:,:,:,j]) ** 2))
        sum_mtr = torch.mean(sum_mtr.view(-1))

        return sum_mtr
