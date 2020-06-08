import pytest
import torch
from asteroid.masknn import recurrent as rec


@pytest.mark.parametrize("mask_act", ["relu", "softmax"])
@pytest.mark.parametrize("out_chan", [None, 10])
@pytest.mark.parametrize("hop_size", [None, 5])
def test_dprnn(mask_act, out_chan, hop_size):
    in_chan, n_src = 20, 2
    model = rec.DPRNN(in_chan=in_chan, n_src=n_src, mask_act=mask_act,
                      chunk_size=20, n_repeats=2, bn_chan=10, hid_size=11,
                      out_chan=out_chan, hop_size=hop_size)
    batch, n_frames = 2, 78
    inp = torch.randn(batch, in_chan, n_frames)
    out = model(inp)
    _ = model.get_config()
    out_chan = out_chan if out_chan else in_chan
    assert out.shape == (batch, n_src, out_chan, n_frames)


@pytest.mark.parametrize("rnn_type", ["LSTM", "GRU", "RNN"])
@pytest.mark.parametrize("dropout", [0., 0.2])
@pytest.mark.parametrize("bidir", [False])
def test_res_rnn(rnn_type, dropout, bidir):
    n_units, n_layers = 20, 3
    model = rec.StackedResidualRNN(rnn_type, n_units, n_layers=n_layers,
                                   dropout=dropout, bidirectional=bidir)
    batch, n_frames = 2, 78
    inp = torch.randn(batch, n_frames, n_units)
    out = model(inp)
    assert out.shape == (batch, n_frames, n_units)


@pytest.mark.parametrize("rnn_type", ["LSTM", "GRU", "RNN"])
@pytest.mark.parametrize("dropout", [0., 0.2])
def test_res_birnn(rnn_type, dropout):
    n_units, n_layers = 20, 3
    model = rec.StackedResidualBiRNN(rnn_type, n_units, n_layers=n_layers,
                                     dropout=dropout, bidirectional=True)
    batch, n_frames = 2, 78
    inp = torch.randn(batch, n_frames, n_units)
    out = model(inp)
    assert out.shape == (batch, n_frames, 2 * n_units)
