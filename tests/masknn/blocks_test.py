import pytest
import torch
from asteroid.masknn import blocks


@pytest.mark.parametrize("mask_act", ["relu", "softmax"])
@pytest.mark.parametrize("out_chan", [None, 10])
@pytest.mark.parametrize("skip_chan", [0, 12])
def test_tdconvnet(mask_act, out_chan, skip_chan):
    in_chan, n_src = 20, 2
    model = blocks.TDConvNet(in_chan=in_chan, n_src=n_src, mask_act=mask_act,
                             n_blocks=2, n_repeats=2, bn_chan=10, hid_chan=11,
                             skip_chan=skip_chan, out_chan=out_chan)
    batch, n_frames = 2, 24
    inp = torch.randn(batch, in_chan, n_frames)
    out = model(inp)
    _ = model.get_config()
    out_chan = out_chan if out_chan else in_chan
    assert out.shape == (batch, n_src, out_chan, n_frames)


@pytest.mark.parametrize("mask_act", ["relu", "softmax"])
@pytest.mark.parametrize("out_chan", [None, 10])
@pytest.mark.parametrize("hop_size", [None, 5])
def test_dprnn(mask_act, out_chan, hop_size):
    in_chan, n_src = 20, 2
    model = blocks.DPRNN(in_chan=in_chan, n_src=n_src, mask_act=mask_act,
                         chunk_size=20, n_repeats=2, bn_chan=10, hid_size=11,
                         out_chan=out_chan, hop_size=hop_size)
    batch, n_frames = 2, 78
    inp = torch.randn(batch, in_chan, n_frames)
    out = model(inp)
    _ = model.get_config()
    out_chan = out_chan if out_chan else in_chan
    assert out.shape == (batch, n_src, out_chan, n_frames)

    
@pytest.mark.parametrize("embed_dim", [10, 20, 30])
def test_chimerapp(embed_dim):
    in_chan, n_src = 52, 2
    model = blocks.ChimeraPP(in_chan, n_src, embedding_dim=embed_dim,
                             hidden_size=50)
    batch, freq_dim, nframes = 10, in_chan, 10
    inp = torch.randn(batch, in_chan, nframes)
    out = model(inp)
    assert out[0].shape == (batch, freq_dim*nframes, embed_dim)
    assert out[1].shape == (batch, n_src, in_chan, nframes)

    
@pytest.mark.parametrize("rnn_type", ["LSTM", "GRU", "RNN"])
@pytest.mark.parametrize("dropout", [0., 0.2])
@pytest.mark.parametrize("bidir", [False])
def test_res_rnn(rnn_type, dropout, bidir):
    n_units, n_layers = 20, 3
    model = blocks.StackedResidualRNN(rnn_type, n_units, n_layers=n_layers,
                                      dropout=dropout, bidirectional=bidir)
    batch, n_frames = 2, 78
    inp = torch.randn(batch, n_frames, n_units)
    out = model(inp)
    assert out.shape == (batch, n_frames, n_units)
