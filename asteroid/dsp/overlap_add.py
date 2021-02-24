import torch
from torch import nn
from ..losses.pit_wrapper import PITReorder


class LambdaOverlapAdd(torch.nn.Module):
    """Overlap-add with lambda transform on segments (not scriptable).

    Segment input signal, apply lambda function (a neural network for example)
    and combine with OLA.

    `LambdaOverlapAdd` can be used with :mod:`asteroid.separate` and the
    `asteroid-infer` CLI.

    Args:
        nnet (callable): Function to apply to each segment.
        n_src (Optional[int]): Number of sources in the output of nnet.
            If None, the number of sources is determined by the network's output,
            but some correctness checks cannot be performed.
        window_size (int): Size of segmenting window.
        hop_size (int): Segmentation hop size.
        window (str): Name of the window (see scipy.signal.get_window) used
            for the synthesis.
        reorder_chunks (bool): Whether to reorder each consecutive segment.
            This might be useful when `nnet` is permutation invariant, as
            source assignements might change output channel from one segment
            to the next (in classic speech separation for example).
            Reordering is performed based on the correlation between
            the overlapped part of consecutive segment.

     Examples
        >>> from asteroid import ConvTasNet
        >>> nnet = ConvTasNet(n_src=2)
        >>> continuous_nnet = LambdaOverlapAdd(
        >>>     nnet=nnet,
        >>>     n_src=2,
        >>>     window_size=64000,
        >>>     hop_size=None,
        >>>     window="hanning",
        >>>     reorder_chunks=True,
        >>>     enable_grad=False,
        >>> )

        >>> # Process wav tensor:
        >>> wav = torch.randn(1, 1, 500000)
        >>> out_wavs = continuous_nnet.forward(wav)
        >>> # asteroid.separate.Separatable support:
        >>> from asteroid.separate import file_separate
        >>> file_separate(continuous_nnet, "example.wav")
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
        self.in_channels = getattr(nnet, "in_channels", None)

        if window:
            from scipy.signal import get_window  # for torch.hub

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
        unfolded = torch.nn.functional.unfold(
            x.unsqueeze(-1),
            kernel_size=(self.window_size, 1),
            padding=(self.window_size, 0),
            stride=(self.hop_size, 1),
        )

        out = []
        n_chunks = unfolded.shape[-1]
        for frame_idx in range(n_chunks):  # for loop to spare memory
            frame = self.nnet(unfolded[..., frame_idx])
            # user must handle multichannel by reshaping to batch
            if frame_idx == 0:
                assert frame.ndim == 3, "nnet should return (batch, n_src, time)"
                if self.n_src is not None:
                    assert frame.shape[1] == self.n_src, "nnet should return (batch, n_src, time)"
                n_src = frame.shape[1]
            frame = frame.reshape(batch * n_src, -1)

            if frame_idx != 0 and self.reorder_chunks:
                # we determine best perm based on xcorr with previous sources
                frame = _reorder_sources(frame, out[-1], n_src, self.window_size, self.hop_size)

            if self.use_window:
                frame = frame * self.window
            else:
                frame = frame / (self.window_size / self.hop_size)
            out.append(frame)

        out = torch.stack(out).reshape(n_chunks, batch * n_src, self.window_size)
        out = out.permute(1, 2, 0)

        out = torch.nn.functional.fold(
            out,
            (n_frames, 1),
            kernel_size=(self.window_size, 1),
            padding=(self.window_size, 0),
            stride=(self.hop_size, 1),
        )
        return out.squeeze(-1).reshape(batch, n_src, -1)

    def forward(self, x):
        """Forward module: segment signal, apply func, combine with OLA.

        Args:
            x (:class:`torch.Tensor`): waveform signal of shape (batch, 1, time).

        Returns:
            :class:`torch.Tensor`: The output of the lambda OLA.
        """
        # Here we can do the reshaping
        with torch.autograd.set_grad_enabled(self.enable_grad):
            olad = self.ola_forward(x)
            return olad

    # Implement `asteroid.separate.Separatable` (separation support)

    @property
    def sample_rate(self):
        return self.nnet.sample_rate

    def _separate(self, wav, *args, **kwargs):
        return self.forward(wav, *args, **kwargs)


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

    """
    batch, frames = current.size()
    current = current.reshape(-1, n_src, frames)
    previous = previous.reshape(-1, n_src, frames)

    overlap_f = window_size - hop_size

    def reorder_func(x, y):
        x = x[..., :overlap_f]
        y = y[..., -overlap_f:]
        # Mean normalization
        x = x - x.mean(-1, keepdim=True)
        y = y - y.mean(-1, keepdim=True)
        # Negative mean Correlation
        return -torch.sum(x.unsqueeze(1) * y.unsqueeze(2), dim=-1)

    # We maximize correlation-like between previous and current.
    pit = PITReorder(reorder_func)
    current = pit(current, previous)
    return current.reshape(batch, frames)


class DualPathProcessing(nn.Module):
    """
    Perform Dual-Path processing via overlap-add as in DPRNN [1].

    Args:
        chunk_size (int): Size of segmenting window.
        hop_size (int): segmentation hop size.

    References
        [1] Yi Luo, Zhuo Chen and Takuya Yoshioka. "Dual-path RNN: efficient
        long sequence modeling for time-domain single-channel speech separation"
        https://arxiv.org/abs/1910.06379
    """

    def __init__(self, chunk_size, hop_size):
        super(DualPathProcessing, self).__init__()
        self.chunk_size = chunk_size
        self.hop_size = hop_size
        self.n_orig_frames = None

    def unfold(self, x):
        r"""
        Unfold the feature tensor from $(batch, channels, time)$ to
        $(batch, channels, chunksize, nchunks)$.

        Args:
            x (:class:`torch.Tensor`): feature tensor of shape $(batch, channels, time)$.

        Returns:
            :class:`torch.Tensor`: spliced feature tensor of shape
            $(batch, channels, chunksize, nchunks)$.

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
        r"""
        Folds back the spliced feature tensor.
        Input shape $(batch, channels, chunksize, nchunks)$ to original shape
        $(batch, channels, time)$ using overlap-add.

        Args:
            x (:class:`torch.Tensor`): spliced feature tensor of shape
                $(batch, channels, chunksize, nchunks)$.
            output_size (int, optional): sequence length of original feature tensor.
                If None, the original length cached by the previous call of
                :meth:`unfold` will be used.

        Returns:
            :class:`torch.Tensor`:  feature tensor of shape $(batch, channels, time)$.

        .. note:: `fold` caches the original length of the input.

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

        # force float div for torch jit
        x /= float(self.chunk_size) / self.hop_size

        return x.reshape(batch, chan, self.n_orig_frames)

    @staticmethod
    def intra_process(x, module):
        r"""Performs intra-chunk processing.

        Args:
            x (:class:`torch.Tensor`): spliced feature tensor of shape
                (batch, channels, chunk_size, n_chunks).
            module (:class:`torch.nn.Module`): module one wish to apply to each chunk
                of the spliced feature tensor.

        Returns:
            :class:`torch.Tensor`: processed spliced feature tensor of shape
            $(batch, channels, chunksize, nchunks)$.

        .. note:: the module should have the channel first convention and accept
            a 3D tensor of shape $(batch, channels, time)$.
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
        r"""Performs inter-chunk processing.

        Args:
            x (:class:`torch.Tensor`): spliced feature tensor of shape
                $(batch, channels, chunksize, nchunks)$.
            module (:class:`torch.nn.Module`): module one wish to apply between
                each chunk of the spliced feature tensor.


        Returns:
            x (:class:`torch.Tensor`): processed spliced feature tensor of shape
            $(batch, channels, chunksize, nchunks)$.

        .. note:: the module should have the channel first convention and accept
            a 3D tensor of shape $(batch, channels, time)$.
        """

        batch, channels, chunk_size, n_chunks = x.size()
        x = x.transpose(1, 2).reshape(batch * chunk_size, channels, n_chunks)
        x = module(x)
        x = x.reshape(batch, chunk_size, channels, n_chunks).transpose(1, 2)
        return x
