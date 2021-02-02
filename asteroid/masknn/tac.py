import torch
import torch.nn as nn
from asteroid.masknn import activations, norms


class TAC(nn.Module):
    """
    Transform-Average-Concatenate inter-microphone-channel permutation invariant communication block.
    From [1].
    Note: supports inputs of shape (batch, mic_channels, features, chunk_size, n_chunks) as in FasNet-TAC.
    The operations are applied for each element in chunk_size and n_chunks.
    Output is of same shape as input.

    Args:
        input_dim (int): Number of features of input representation.
        hidden_dim (int, optional): size of hidden layers in TAC operations.
        activation (str, optional): type of activation used. See asteroid.masknn.activations.
        norm_type (str, optional): type of normalization layer used. See asteroid.masknn.norms.


    [1] Luo, Yi, et al. "End-to-end microphone permutation and number invariant multi-channel speech separation."
    ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2020.
    """

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
        # input is 5D because it is multi-channel DPRNN.
        # DPRNN single channel is 4D.

        batch_size, nmics, channels, chunk_size, n_chunks = x.size()
        # first operation: transform the input for each frame and independently on each mic channel.
        output = self.input_tf(
            x.permute(0, 3, 4, 1, 2).reshape(batch_size * nmics * chunk_size * n_chunks, channels)
        ).reshape(batch_size, chunk_size, n_chunks, nmics, self.hidden_dim)

        # mean pooling across channels
        if valid_mics.max() == 0:
            # fixed geometry array
            mics_mean = output.mean(1)
        else:
            # only consider valid channels in each batch element: each example can have different number of microphones.
            mics_mean = [
                output[b, :, :, : valid_mics[b]].mean(2).unsqueeze(0) for b in range(batch_size)
            ]  # 1, dim1*dim2, H
            mics_mean = torch.cat(mics_mean, 0)  # B*dim1*dim2, H

        # the average is processed by a non-linear transform
        mics_mean = self.avg_tf(
            mics_mean.reshape(batch_size * chunk_size * n_chunks, self.hidden_dim)
        )
        mics_mean = (
            mics_mean.reshape(batch_size, chunk_size, n_chunks, self.hidden_dim)
            .unsqueeze(3)
            .expand_as(output)
        )

        # concatenate the transformed average in each channel with the original feats and project back to same number of features
        output = torch.cat([output, mics_mean], -1)
        output = self.concat_tf(
            output.reshape(batch_size * chunk_size * n_chunks * nmics, -1)
        ).reshape(batch_size, chunk_size, n_chunks, nmics, -1)
        output = self.norm(
            output.permute(0, 3, 4, 1, 2).reshape(batch_size * nmics, -1, chunk_size, n_chunks)
        ).reshape(batch_size, nmics, -1, chunk_size, n_chunks)

        return output + x
