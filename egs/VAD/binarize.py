from itertools import groupby
import torch


class Binarize(torch.nn.Module):
    def __init__(self, threshold=0.5, stability=0.1, sample_rate=8000):
        super().__init__()
        self.threshold = threshold
        self.stability = stability
        self.sample_rate = sample_rate

    def forward(self, x):
        active = x > self.threshold
        active = active.squeeze(1).tolist()
        pairs = count_same_pair(active)
        active = fit(pairs, self.stability, self.sample_rate)
        return active


def count_same_pair(nums):
    result = []
    for num in nums:
        result.append([[i, sum(1 for _ in group)] for i, group in groupby(num)])
    return result


def fit(pairs, stability, sample_rate):
    batch_active = []
    for pair in pairs:
        active = []
        if len(pair) == 1:
            if pair[0][0]:
                active = torch.ones(pair[0][1])
            else:
                active = torch.zeros(pair[0][1])
            return active

        i = 0
        while i < len(pair):
            actived = 0
            not_actived = 0
            if pair[i][1] < int(stability * sample_rate):
                while i < len(pair) and pair[i][1] < int(stability * sample_rate):
                    if pair[i][0]:
                        actived += pair[i][1]
                        i += 1
                    else:
                        not_actived += pair[i][1]
                        i += 1
                if actived + not_actived < int(stability * sample_rate) and len(active) > 0:
                    if active[-1][0] == 1:
                        active.append(torch.ones(actived + not_actived))
                    else:
                        active.append(torch.zeros(actived + not_actived))
                elif actived + not_actived < int(stability * sample_rate) and len(active) == 0:
                    active.append(torch.zeros(actived + not_actived))
                else:
                    if actived > not_actived:
                        active.append(torch.ones(actived + not_actived))
                    else:
                        active.append(torch.zeros(actived + not_actived))

            else:
                if pair[i][0]:
                    active.append(torch.ones(pair[i][1]))
                else:
                    active.append(torch.zeros(pair[i][1]))
                i += 1

        batch_active.append(torch.hstack(active))
    batch_active = torch.vstack(batch_active).unsqueeze(1)
    return batch_active
