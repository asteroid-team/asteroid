import torch
import torch.nn as nn
from asteroid.masknn import activations, norms


class TAC(nn.Module):
    def __init__(self, input_dim, hidden_dim=384, activation="prelu", norm_type="gLN"):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.input_tf = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), activations.get(activation)()
        )
        self.avg_tf = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), activations.get(activation)()
        )
        self.concat_tf = nn.Sequential(
            nn.Linear(2 * hidden_dim, input_dim), activations.get(activation)()
        )
        self.norm = norms.get(norm_type)(input_dim)

    def forward(self, x, valid_mics):

        batch_size, nmics, channels, chunk_size, n_chunks = x.size()
        output = self.input_tf(
            x.permute(0, 3, 4, 1, 2).reshape(batch_size * nmics * chunk_size * n_chunks, channels)
        ).reshape(batch_size, chunk_size, n_chunks, nmics, self.hidden_dim)

        # mean pooling across channels
        if valid_mics.max() == 0:
            # fixed geometry array
            mics_mean = output.mean(1)
        else:
            # only consider valid channels
            mics_mean = [
                output[b, :, :, : valid_mics[b]].mean(2).unsqueeze(0) for b in range(batch_size)
            ]  # 1, dim1*dim2, H
            mics_mean = torch.cat(mics_mean, 0)  # B*dim1*dim2, H

        mics_mean = self.avg_tf(
            mics_mean.reshape(batch_size * chunk_size * n_chunks, self.hidden_dim)
        )
        mics_mean = (
            mics_mean.reshape(batch_size, chunk_size, n_chunks, self.hidden_dim)
            .unsqueeze(3)
            .expand_as(output)
        )
        output = torch.cat([output, mics_mean], -1)
        output = self.concat_tf(
            output.reshape(batch_size * chunk_size * n_chunks * nmics, -1)
        ).reshape(batch_size, chunk_size, n_chunks, nmics, -1)
        output = self.norm(
            output.permute(0, 3, 4, 1, 2).reshape(batch_size * nmics, -1, chunk_size, n_chunks)
        ).reshape(batch_size, nmics, -1, chunk_size, n_chunks)

        return output + x