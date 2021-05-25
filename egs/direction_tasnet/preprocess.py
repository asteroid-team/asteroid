from torch import nn
from conv_stft import STFT
from torch.autograd import Variable
from deepbeam import R_global
import numpy as np
import torch
sr = 48000
n_mic = 6
filter_length = int(sr * 0.004)
hop_length = int(sr * 0.00125)  # doesn't need to be specified. if not specified, it's the same as filter_length
window = 'hann'
win_length = int(sr * 0.0025)
stft = STFT(filter_length=filter_length, hop_length=hop_length, win_length=win_length,
            window=window)



m_total = 97
frequency_vector = np.linspace(0, sr//2, m_total)
n_grid = 36
V = 343
z = np.zeros((n_mic, n_grid, m_total))
a = np.arange(m_total).reshape(1, 1, m_total)
m_data = torch.from_numpy(z + a)
n_sp = 3




def Prep(data):
        dic_data = isinstance(data, dict)
        if dic_data:
            angle = torch.tensor(data["angle"]).unsqueeze(dim=0)
            input = torch.from_numpy(data["mix"]).unsqueeze(dim=0).float()
            input = torch.transpose(input, 2, 1)
            data["mix"] = input.squeeze()
            R = data["R"]
        else:
            input = torch.from_numpy(data[0]).unsqueeze(dim=0).float()
            angle = torch.tensor(data[2]).unsqueeze(dim=0)
            R = data[5]
            return_list = []

        mic_array_layout = R - np.tile(R[:, 0].reshape((3, 1)), (1, n_mic))
        pairs = ((0, 3), (1, 4), (2, 5), (0, 1), (2, 3), (4, 5))
        ori_pairs = ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5))
        delay = np.zeros((n_mic, n_grid))
        for h, m in enumerate(ori_pairs):
            dx = mic_array_layout[0, m[1]] - mic_array_layout[0, m[0]]
            dy = mic_array_layout[1, m[1]] - mic_array_layout[1, m[0]]
            for i in range(n_grid):
                delay[h, i] = dx * np.cos(i * np.pi / 18) + dy * np.sin(i * np.pi / 18)
        delay = torch.from_numpy(delay).unsqueeze(dim=-1).expand(-1, -1, m_total)
        w = torch.exp(-2j * np.pi * m_data * delay) / V
        batch_size = input.size(0)
        mag, ph, real, image = stft.transform(input.reshape(-1, input.size()[-1]))



        pad = Variable(torch.zeros(mag.size()[0], mag.size()[1], 1)).type(input.type())
        mag = torch.cat([mag, pad], -1)
        ph = torch.cat([ph, pad], -1)
        channel = mag.size()[-1]
        mag = mag.view(batch_size, n_mic, -1, channel)
        ph = ph.view(batch_size, n_mic, -1, channel)
        #LPS = 10 * torch.log10(mag ** 2 + 10e-20)
        complex = (mag * torch.exp(ph * 1j))
        IPD_list = []
        for m in pairs:
            com_u1 = complex[:, m[0]]
            com_u2 = complex[:, m[1]]
            IPD = torch.angle(com_u1) - torch.angle(com_u2)
            #IPD /= (frequency_vector + 1.0)[:, None]
            #IPD = IPD % (2 * np.pi)
            IPD = IPD.unsqueeze(dim=1)
            IPD_list.append(IPD)
        IPD = torch.cat(IPD_list, dim=1)
        complex = complex.unsqueeze(dim=2).expand(-1, -1, n_grid, -1, -1)
        for i in range(n_sp):
            ang = angle[:, i]
            steering_vector = __get_steering_vector(ang, pairs, mic_array_layout)
            steering_vector = steering_vector.unsqueeze(dim=-1)
            AF = steering_vector * torch.exp(1j * IPD)
            AF = AF/(torch.sqrt(AF.real ** 2 + AF.imag**2) + 10e-20)
            AF = AF.sum(dim=1)
            w_ = w.unsqueeze(dim=0).expand(AF.size()[0], -1, -1, -1).unsqueeze(-1).expand(-1, -1, -1, -1, channel)
            mod_w_com = (w_ * complex) * torch.conj(w_ * complex)
            dpr = mod_w_com.sum(dim=1) / ((mod_w_com).sum(dim=1).sum(dim=1, keepdims=True) + 10e-20)
            p = (ang/np.pi*18).type(torch.long)
            dpr = dpr[range(batch_size), p]
            feature_IPD = IPD.reshape(batch_size, IPD.size()[1] * IPD.size()[2], IPD.size(-1))
            feature_list = [AF, torch.cos(feature_IPD), dpr]
            fusion = torch.cat(feature_list, dim=1).real.float()
            if dic_data:
               data[i] = fusion.squeeze()
            else:
               return_list.append(fusion)
        if not dic_data:
            return return_list


def __get_steering_vector(angle, pairs, mic_array_layout):
        steering_vector = np.zeros((len(angle), len(frequency_vector), 6), dtype='complex')

        # get delay
        delay = np.zeros((len(angle), n_mic))
        for h, m in enumerate(pairs):
            dx = mic_array_layout[0, m[1]] - mic_array_layout[0, m[0]]
            dy = mic_array_layout[1, m[1]] - mic_array_layout[1, m[0]]
            delay[:, h] = dx * np.cos(angle) + dy * np.sin(angle)

        for f, frequency in enumerate(frequency_vector):
            for m in range(len(pairs)):
                steering_vector[:, f, m] = np.exp(1j * 2 * np.pi * frequency * delay[:, m] / V)
        steering_vector = torch.from_numpy(steering_vector)

        return torch.transpose(steering_vector, 1, 2)


if __name__ == "__main__":
    pass
    import numpy as np

    data = {"mix":np.random.rand(36000, 6), "angle":[2.4, 2.5, 3.1]}

    Prep(data)