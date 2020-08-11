import torch
from scipy.signal import get_window
from asteroid.losses import PITLossWrapper


class OverlapAddWrapper(torch.nn.Module):
    def __init__(self, nnet, n_src, window_size, hop_size, window="hanning", reorder_chunks=True):
        super(OverlapAddWrapper, self).__init__()

        assert window_size % 2 == 0, "Window size must be even"

        self.nnet = nnet
        self.window_size = window_size
        self.hop_size = hop_size
        self.n_src = n_src

        if window:
            window = get_window(window, self.window_size).astype("float32")
            window = torch.from_numpy(window)
        else:
            window = torch.ones(self.window_size)

        self.register_buffer("window", window)
        self.reorder_chunks = reorder_chunks

    def _reorder_sources(self, current, previous):
        # we compute xcorr for all permutations
        # we get index for min
        batch, frames = current.size()
        current = current.reshape(-1, self.n_src, frames)
        previous = previous.reshape(-1, self.n_src, frames)

        overlap_f = self.window_size - self.hop_size
        pw_losses = PITLossWrapper.get_pw_losses(lambda x, y: torch.sum((x.unsqueeze(1) * y.unsqueeze(2))),
                                                 current[..., :overlap_f],
                                                 previous[..., -overlap_f:])
        _, perms = PITLossWrapper.find_best_perm(pw_losses, self.n_src)
        current = PITLossWrapper.reorder_source(current, self.n_src, perms)
        return current.reshape(batch, frames)

    def forward(self, x):
        with torch.no_grad():
            assert len(x.shape) == 3

            # Overlap and add:
            # [batch, channels, n_frames] -> [batch, channels, window_size, n_chunks]
            batch, channels, n_frames = x.size()
            folded = torch.nn.functional.unfold(x.unsqueeze(-1), kernel_size=(
            self.window_size, 1),
            padding=(self.window_size, 0),
            stride=(self.hop_size, 1))

            out = []
            n_chunks = folded.shape[-1]
            for f in range(n_chunks):  # for loop to spare memory
                tmp = self.nnet(folded[..., f])
                # user must handle multichannel by reshaping to batch
                assert len(tmp.size()) == 3
                assert tmp.shape[1] == self.n_src
                tmp = tmp.reshape(batch * self.n_src, -1)

                if f != 0 and self.reorder_chunks:
                    # we determine best perm based on xcorr with previous sources
                    tmp = self._reorder_sources(tmp, out[-1])

                tmp = tmp*self.window
                out.append(tmp)

            out = torch.stack(out).reshape(n_chunks, batch*self.n_src, self.window_size).permute(1, 2, 0)
            out = torch.nn.functional.fold(out, (n_frames, 1),
                                           kernel_size=(self.window_size, 1),
                                           padding=(self.window_size, 0),
                                           stride=(self.hop_size, 1))

            return out.squeeze(-1).reshape(batch, self.n_src, -1)