import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utility import models, sdr


# Conv-TasNet
class TasNet(nn.Module):
    def __init__(self, enc_dim=97, feature_dim=128, sr=48000, win=2.5, layer=8, stack=3,
                 kernel=3, num_spk=1, causal=True):
        super(TasNet, self).__init__()
        
        # hyper parameters
        self.num_spk = num_spk

        self.enc_dim = enc_dim
        self.feature_dim = feature_dim
        
        self.win = int(sr*win/1000)
        self.stride = self.win // 2
        
        self.layer = layer
        self.stack = stack
        self.kernel = kernel

        self.causal = causal
        self.channel = 1
        
        # input encoder
        self.encoder = nn.Conv1d(self.channel, self.enc_dim, self.win, bias=False, stride=self.stride)
        
        # TCN separator
        self.TCN = models.TCN(self.enc_dim * 9, self.enc_dim*self.num_spk, self.feature_dim, self.feature_dim*4,
                              self.layer, self.stack, self.kernel, causal=self.causal)

        self.receptive_field = self.TCN.receptive_field
        
        # output decoder
        self.decoder = nn.ConvTranspose1d(self.enc_dim, 1, self.win, bias=False, stride=self.stride)

    def pad_signal(self, input):

        # input is the waveforms: (B, T) or (B, 1, T)
        # reshape and padding
        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")
        
        if input.dim() == 2:
            input = input.unsqueeze(1)
        batch_size = input.size(0)
        nsample = input.size(2)
        rest = self.win - (self.stride + nsample % self.win) % self.win
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, 6, rest)).type(input.type())
            input = torch.cat([input, pad], 2)
        
        pad_aux = Variable(torch.zeros(batch_size, 6, self.stride)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest
        
    def forward(self, input, feature):
        
        # padding
        output, rest = self.pad_signal(input)

        batch_size = output.size(0)
        
        # waveform encoder
        enc = self.encoder(output[:, :1])  # B, N, L
        enc_output = torch.cat([enc, feature], dim=1)

        # generate masks
        masks = torch.sigmoid(self.TCN(enc_output)).view(batch_size, self.num_spk, self.enc_dim, -1)  # B, C, N, L
        masked_output = enc.unsqueeze(1) * masks  # B, C, N, L
        
        # waveform decoder
        output = self.decoder(masked_output.view(batch_size*self.num_spk, self.enc_dim, -1))  # B*C, 1, L
        output = output[:,:,self.stride:-(rest+self.stride)].contiguous()  # B*C, 1, L
        output = output.view(batch_size, self.num_spk, -1)  # B, C, T
        
        return output


def test_conv_tasnet():
    x = torch.rand(16, 6, 32000)
    nnet = TasNet()
    pytorch_total_params = sum(p.numel() for p in nnet.parameters() if p.requires_grad)
    x = nnet(x)
    print(x.size())

if __name__ == "__main__":
    test_conv_tasnet()
