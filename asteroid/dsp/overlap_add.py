import torch
from scipy.signal import get_window
from asteroid.losses import PITLossWrapper


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
