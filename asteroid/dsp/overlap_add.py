import torch
from scipy.signal import get_window
from asteroid.losses import PITLossWrapper
from torch import nn


class LambdaOverlapAdd(torch.nn.Module):
    """ Segment signal, apply func, combine with OLA.

    Args:
        nnet (callable): function to apply to each segment.
        n_src (int): Number of sources in the output of nnet.
        window_size (int): Size of segmenting window.
        hop_size (int): segmentation hop size.
        window (str): Name of the window (see scipy.signal.get_window)
        reorder_chunks (bool): whether to reorder each consecutive segment.
    """

    def __init__(
        self,
        nnet,
        n_src,
        window_size,
        hop_size=None,
        window="hanning",
        reorder_chunks=True,
        enable_grad=False,
    ):
        super().__init__()
        assert window_size % 2 == 0, "Window size must be even"

        self.nnet = nnet
        self.window_size = window_size
        self.hop_size = hop_size if hop_size is not None else window_size // 2
        self.n_src = n_src

        if window:
            window = get_window(window, self.window_size).astype("float32")
            window = torch.from_numpy(window)
            self.use_window = True
        else:
            self.use_window = False

        self.register_buffer("window", window)
        self.reorder_chunks = reorder_chunks
        self.enable_grad = enable_grad

    def ola_forward(self, x):
        """Heart of the class: segment signal, apply func, combine with OLA."""

        assert x.ndim == 3

        batch, channels, n_frames = x.size()
        # Overlap and add:
        # [batch, chans, n_frames] -> [batch, chans, win_size, n_chunks]
        folded = torch.nn.functional.unfold(
            x.unsqueeze(-1),
            kernel_size=(self.window_size, 1),
            padding=(self.window_size, 0),
            stride=(self.hop_size, 1),
        )

        out = []
        n_chunks = folded.shape[-1]
        for frame_idx in range(n_chunks):  # for loop to spare memory
            tmp = self.nnet(folded[..., frame_idx])
            # user must handle multichannel by reshaping to batch
            if frame_idx == 0:
                assert tmp.ndim == 3, "nnet should return (batch, n_src, time)"
                assert tmp.shape[1] == self.n_src, "nnet should return (batch, n_src, time)"
            tmp = tmp.reshape(batch * self.n_src, -1)

            if frame_idx != 0 and self.reorder_chunks:
                # we determine best perm based on xcorr with previous sources
                tmp = _reorder_sources(tmp, out[-1], self.n_src, self.window_size, self.hop_size)

            if self.use_window:
                tmp = tmp * self.window
            else:
                tmp = tmp / (self.window_size / self.hop_size)
            out.append(tmp)

        out = torch.stack(out).reshape(n_chunks, batch * self.n_src, self.window_size)
        out = out.permute(1, 2, 0)

        out = torch.nn.functional.fold(
            out,
            (n_frames, 1),
            kernel_size=(self.window_size, 1),
            padding=(self.window_size, 0),
            stride=(self.hop_size, 1),
        )
        return out.squeeze(-1).reshape(batch, self.n_src, -1)

    def forward(self, x):
        """ Forward module: segment signal, apply func, combine with OLA.

        Args:
            x (:class:`torch.Tensor`): waveform signal of shape (batch, 1, time).

        Returns:
            :class:`torch.Tensor`: The output of the lambda OLA.
        """
        # Here we can do the reshaping
        with torch.autograd.set_grad_enabled(self.enable_grad):
            olad = self.ola_forward(x)
            return olad


def _reorder_sources(
    current: torch.FloatTensor,
    previous: torch.FloatTensor,
    n_src: int,
    window_size: int,
    hop_size: int,
):
    """
     Reorder sources in current chunk to maximize correlation with previous chunk.
     Used for Continuous Source Separation. Standard dsp correlation is used
     for reordering.


    Args:
        current (:class:`torch.Tensor`): current chunk, tensor
                                        of shape (batch, n_src, window_size)
        previous (:class:`torch.Tensor`): previous chunk, tensor
                                        of shape (batch, n_src, window_size)
        n_src (:class:`int`): number of sources.
        window_size (:class:`int`): window_size, equal to last dimension of
                                    both current and previous.
        hop_size (:class:`int`): hop_size between current and previous tensors.

    Returns:
        current:

    """
    batch, frames = current.size()
    current = current.reshape(-1, n_src, frames)
    previous = previous.reshape(-1, n_src, frames)

    overlap_f = window_size - hop_size
    pw_losses = PITLossWrapper.get_pw_losses(
        lambda x, y: torch.sum((x.unsqueeze(1) * y.unsqueeze(2))),
        current[..., :overlap_f],
        previous[..., -overlap_f:],
    )
    _, perms = PITLossWrapper.find_best_perm(pw_losses, n_src)
    current = PITLossWrapper.reorder_source(current, n_src, perms)
    return current.reshape(batch, frames)


class DualPathProcessing(nn.Module):
    """ Perform Dual-Path processing via overlap-add as in DPRNN [1].

     Args:
        chunk_size (int): Size of segmenting window.
        hop_size (int): segmentation hop size.

    References:
        [1] "Dual-path RNN: efficient long sequence modeling for
            time-domain single-channel speech separation", Yi Luo, Zhuo Chen
            and Takuya Yoshioka. https://arxiv.org/abs/1910.06379
    """

    def __init__(self, chunk_size, hop_size):
        super(DualPathProcessing, self).__init__()
        self.chunk_size = chunk_size
        self.hop_size = hop_size
        self.n_orig_frames = None

    def unfold(self, x):
        """ Unfold the feature tensor from

        (batch, channels, time) to (batch, channels, chunk_size, n_chunks).

        Args:
            x: (:class:`torch.Tensor`): feature tensor of shape (batch, channels, time).

        Returns:
            x: (:class:`torch.Tensor`): spliced feature tensor of shape
                (batch, channels, chunk_size, n_chunks).

        """
        # x is (batch, chan, frames)
        batch, chan, frames = x.size()
        assert x.ndim == 3
        self.n_orig_frames = x.shape[-1]
        unfolded = torch.nn.functional.unfold(
            x.unsqueeze(-1),
            kernel_size=(self.chunk_size, 1),
            padding=(self.chunk_size, 0),
            stride=(self.hop_size, 1),
        )

        return unfolded.reshape(
            batch, chan, self.chunk_size, -1
        )  # (batch, chan, chunk_size, n_chunks)

    def fold(self, x, output_size=None):
        """ Folds back the spliced feature tensor.

        Input shape (batch, channels, chunk_size, n_chunks) to original shape
        (batch, channels, time) using overlap-add.

        Args:
            x: (:class:`torch.Tensor`): spliced feature tensor of shape
                (batch, channels, chunk_size, n_chunks).
            output_size: (int, optional): sequence length of original feature tensor.
                If None, the original length cached by the previous call of `unfold`
                will be used.

        Returns:
            x: (:class:`torch.Tensor`):  feature tensor of shape (batch, channels, time).

        .. note:: `fold` caches the original length of the pr

        """
        output_size = output_size if output_size is not None else self.n_orig_frames
        # x is (batch, chan, chunk_size, n_chunks)
        batch, chan, chunk_size, n_chunks = x.size()
        to_unfold = x.reshape(batch, chan * self.chunk_size, n_chunks)
        x = torch.nn.functional.fold(
            to_unfold,
            (output_size, 1),
            kernel_size=(self.chunk_size, 1),
            padding=(self.chunk_size, 0),
            stride=(self.hop_size, 1),
        )

        x /= self.chunk_size / self.hop_size

        return x.reshape(batch, chan, self.n_orig_frames)

    @staticmethod
    def intra_process(x, module):
        """ Performs intra-chunk processing.

        Args:
            x (:class:`torch.Tensor`): spliced feature tensor of shape
                (batch, channels, chunk_size, n_chunks).
            module (:class:`torch.nn.Module`): module one wish to apply to each chunk
                of the spliced feature tensor.


        Returns:
            x (:class:`torch.Tensor`): processed spliced feature tensor of shape
                (batch, channels, chunk_size, n_chunks).

        .. note:: the module should have the channel first convention and accept
            a 3D tensor of shape (batch, channels, time).
        """

        # x is (batch, channels, chunk_size, n_chunks)
        batch, channels, chunk_size, n_chunks = x.size()
        # we reshape to batch*chunk_size, channels, n_chunks
        x = x.transpose(1, -1).reshape(batch * n_chunks, chunk_size, channels).transpose(1, -1)
        x = module(x)
        x = x.reshape(batch, n_chunks, channels, chunk_size).transpose(1, -1).transpose(1, 2)
        return x

    @staticmethod
    def inter_process(x, module):
        """ Performs inter-chunk processing.

        Args:
            x (:class:`torch.Tensor`): spliced feature tensor of shape
                (batch, channels, chunk_size, n_chunks).
            module (:class:`torch.nn.Module`): module one wish to apply between
                each chunk of the spliced feature tensor.


        Returns:
            x (:class:`torch.Tensor`): processed spliced feature tensor of shape
                (batch, channels, chunk_size, n_chunks).

        .. note:: the module should have the channel first convention and accept
            a 3D tensor of shape (batch, channels, time).
        """

        batch, channels, chunk_size, n_chunks = x.size()
        x = x.transpose(1, 2).reshape(batch * chunk_size, channels, n_chunks)
        x = module(x)
        x = x.reshape(batch, chunk_size, channels, n_chunks).transpose(1, 2)
        return x
