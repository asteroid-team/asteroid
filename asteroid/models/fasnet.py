from torch import nn
import torch
from asteroid.masknn.recurrent import DPRNNBlock
import torch.nn.functional as F
from asteroid.masknn import norms
from .base_models import BaseModel
from asteroid.masknn.tac import TAC


def seq_cos_sim(ref, target, eps=1e-8):  # we may want to move this in DSP
    """
    Cosine similarity between some reference mics and some target mics
    ref: shape (nmic1, L, seg1)
    target: shape (nmic2, L, seg2)
    """

    assert ref.size(1) == target.size(1), "Inputs should have same length."
    assert ref.size(2) >= target.size(
        2
    ), "Reference input should be no smaller than the target input."

    seq_length = ref.size(1)

    larger_ch = ref.size(0)
    if target.size(0) > ref.size(0):
        ref = ref.expand(target.size(0), ref.size(1), ref.size(2)).contiguous()  # nmic2, L, seg1
        larger_ch = target.size(0)
    elif target.size(0) < ref.size(0):
        target = target.expand(
            ref.size(0), target.size(1), target.size(2)
        ).contiguous()  # nmic1, L, seg2

    # L2 norms
    ref_norm = F.conv1d(
        ref.view(1, -1, ref.size(2)).pow(2),
        torch.ones(ref.size(0) * ref.size(1), 1, target.size(2)).type(ref.type()),
        groups=larger_ch * seq_length,
    )  # 1, larger_ch*L, seg1-seg2+1
    ref_norm = ref_norm.sqrt() + eps
    target_norm = target.norm(2, dim=2).view(1, -1, 1) + eps  # 1, larger_ch*L, 1
    # cosine similarity
    cos_sim = F.conv1d(
        ref.view(1, -1, ref.size(2)),
        target.view(-1, 1, target.size(2)),
        groups=larger_ch * seq_length,
    )  # 1, larger_ch*L, seg1-seg2+1
    cos_sim = cos_sim / (ref_norm * target_norm)

    return cos_sim.view(larger_ch, seq_length, -1)


class FasNetTAC(BaseModel):
    def __init__(
        self,
        enc_dim,
        feature_dim,
        hidden_dim,
        n_layers=4,
        n_src=2,
        window_ms=4,
        stride=None,
        context_ms=16,
        samplerate=16000,
        tac_hidden_dim=384,
        norm_type="gLN",
        chunk_size=50,
        hop_size=25,
        bidirectional=True,
        rnn_type="LSTM",
        dropout=0.0,
        use_tac=True,
    ):
        super().__init__()

        self.enc_dim = enc_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_src = n_src

        assert window_ms % 2 == 0, "Window length should be even"
        # parameters
        self.window_ms = window_ms
        self.context_ms = context_ms
        self.samplerate = samplerate
        self.window = int(samplerate * window_ms / 1000)
        self.context = int(samplerate * context_ms / 1000)
        if not stride:
            self.stride = self.window // 2
        else:
            self.stride = int(samplerate * stride / 1000)
        self.filter_dim = self.context * 2 + 1
        self.output_dim = self.context * 2 + 1
        self.tac_hidden_dim = tac_hidden_dim
        self.norm_type = norm_type
        self.chunk_size = chunk_size
        self.hop_size = hop_size
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        self.dropout = dropout
        self.use_tac = use_tac

        # waveform encoder
        self.encoder = nn.Conv1d(1, self.enc_dim, self.context * 2 + self.window, bias=False)
        self.enc_LN = norms.get(norm_type)(self.enc_dim)

        # DPRNN here + TAC at each layer
        self.bottleneck = nn.Conv1d(self.filter_dim + self.enc_dim, self.feature_dim, 1, bias=False)

        self.DPRNN_TAC = nn.ModuleList([])
        for i in range(self.n_layers):
            tmp = nn.ModuleList(
                [
                    DPRNNBlock(
                        self.enc_dim,
                        self.hidden_dim,
                        norm_type,
                        bidirectional,
                        rnn_type,
                        dropout=dropout,
                    ),
                ]
            )
            if self.use_tac:
                tmp.append(TAC(self.enc_dim, tac_hidden_dim, norm_type=norm_type))
            self.DPRNN_TAC.append(tmp)

        # DPRNN output layers
        self.conv_2D = nn.Sequential(
            nn.PReLU(), nn.Conv2d(self.enc_dim, self.n_src * self.enc_dim, 1)
        )
        self.tanh = nn.Sequential(nn.Conv1d(self.feature_dim, self.output_dim, 1), nn.Tanh())
        self.gate = nn.Sequential(nn.Conv1d(self.feature_dim, self.output_dim, 1), nn.Sigmoid())

    @staticmethod
    def windowing_with_context(x, window, context):
        batch_size, nmic, nsample = x.shape

        unfolded = F.unfold(
            x.unsqueeze(-1),
            kernel_size=(window + 2 * context, 1),
            padding=(context + window, 0),
            stride=(window // 2, 1),
        )

        n_chunks = unfolded.size(-1)
        unfolded = unfolded.reshape(batch_size, nmic, window + 2 * context, n_chunks)

        return (
            unfolded[:, :, context : context + window].transpose(2, -1),
            unfolded.transpose(2, -1),
        )

    def forward(self, x, valid_mics):

        n_samples = x.size(-1)  # original number of samples of multichannel audio
        all_seg, all_mic_context = self.windowing_with_context(x, self.window, self.context)
        batch_size, n_mics, seq_length, feats = all_mic_context.size()
        # all_seg contains only the central window, all_mic_context contains also the right and left context

        # encoder applies a filter on each all_mic_context feats
        enc_output = (
            self.encoder(all_mic_context.reshape(batch_size * n_mics * seq_length, 1, feats))
            .reshape(batch_size * n_mics, seq_length, self.enc_dim)
            .transpose(1, 2)
            .contiguous()
        )  # B*n_mics, seq_len, enc_dim
        enc_output = self.enc_LN(enc_output).reshape(
            batch_size, n_mics, self.enc_dim, seq_length
        )  # apply norm

        # for each context window cosine similarity is computed
        # the first channel is chosen as a reference
        ref_seg = all_seg[:, 0].contiguous().view(1, -1, self.window)
        all_context = (
            all_mic_context.transpose(0, 1)
            .contiguous()
            .view(n_mics, -1, self.context * 2 + self.window)
        )
        all_cos_sim = seq_cos_sim(all_context, ref_seg)
        all_cos_sim = (
            all_cos_sim.view(n_mics, batch_size, seq_length, self.filter_dim)
            .permute(1, 0, 3, 2)
            .contiguous()
        )  # B, nmic, 2*context + 1, seq_len

        # encoder features and cosine similarity features are concatenated
        input_feature = torch.cat([enc_output, all_cos_sim], 2)
        # apply bottleneck to reduce parameters and feed to DPRNN
        input_feature = self.bottleneck(input_feature.reshape(batch_size * n_mics, -1, seq_length))

        # we unfold the features for dual path processing
        unfolded = F.unfold(
            input_feature.unsqueeze(-1),
            kernel_size=(self.chunk_size, 1),
            padding=(self.chunk_size, 0),
            stride=(self.hop_size, 1),
        )
        n_chunks = unfolded.size(-1)
        unfolded = unfolded.reshape(batch_size * n_mics, self.enc_dim, self.chunk_size, n_chunks)

        for i in range(self.n_layers):
            # at each layer we apply for DPRNN to process each mic independently and then TAC for inter-mic processing.
            if self.use_tac:
                dprnn, tac = self.DPRNN_TAC[i]
            else:
                dprnn = self.DPRNN_TAC[i][0]
            unfolded = dprnn(unfolded)
            if self.use_tac:
                b, ch, chunk_size, n_chunks = unfolded.size()
                unfolded = unfolded.reshape(-1, n_mics, ch, chunk_size, n_chunks)
                unfolded = tac(unfolded, valid_mics).reshape(
                    batch_size * n_mics, self.enc_dim, self.chunk_size, n_chunks
                )

        # output, 2D conv to get different feats for each source
        unfolded = self.conv_2D(unfolded).reshape(
            batch_size * n_mics * self.n_src, self.enc_dim * self.chunk_size, n_chunks
        )

        # dual path processing is done we fold back
        folded = F.fold(
            unfolded,
            (seq_length, 1),
            kernel_size=(self.chunk_size, 1),
            padding=(self.chunk_size, 0),
            stride=(self.hop_size, 1),
        )

        # dividing to assure perfect reconstruction
        folded = folded.squeeze(-1) / (self.chunk_size / self.hop_size)

        # apply gating to output and scaling to -1 and 1)
        folded = self.tanh(folded) * self.gate(folded)
        folded = folded.view(batch_size, n_mics, self.n_src, -1, seq_length)

        # beamforming
        # convolving with all mic context --> Filter and Sum
        all_mic_context = all_mic_context.unsqueeze(2).repeat(1, 1, 2, 1, 1)

        all_bf_output = F.conv1d(
            all_mic_context.view(1, -1, self.context * 2 + self.window),
            folded.transpose(3, -1).contiguous().view(-1, 1, self.filter_dim),
            groups=batch_size * n_mics * self.n_src * seq_length,
        )
        all_bf_output = all_bf_output.view(batch_size, n_mics, self.n_src, seq_length, self.window)

        # fold back to obtain signal
        all_bf_output = F.fold(
            all_bf_output.reshape(
                batch_size * n_mics * self.n_src, seq_length, self.window
            ).transpose(1, -1),
            (n_samples, 1),
            kernel_size=(self.window, 1),
            padding=(self.window, 0),
            stride=(self.window // 2, 1),
        )

        bf_signal = all_bf_output.reshape(batch_size, n_mics, self.n_src, n_samples)

        # we sum over mics after filtering (filters will realign the phase and time shift)
        if valid_mics.max() == 0:
            bf_signal = bf_signal.mean(1)
        else:
            bf_signal = [
                bf_signal[b, : valid_mics[b]].mean(0).unsqueeze(0) for b in range(batch_size)
            ]
            bf_signal = torch.cat(bf_signal, 0)

        return bf_signal

    def get_config(self):
        config = {
            "enc_dim": self.enc_dim,
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "n_src": self.n_src,
            "window_ms": self.window_ms,
            "stride": self.stride,
            "context_ms": self.context_ms,
            "samplerate": self.samplerate,
            "tac_hidden_dim": self.tac_hidden_dim,
            "norm_type": self.norm_type,
            "chunk_size": self.chunk_size,
            "hop_size": self.hop_size,
            "bidirectional": self.bidirectional,
            "rnn_type": self.rnn_type,
            "dropout": self.dropout,
            "use_tac": self.use_tac,
        }

        return config
