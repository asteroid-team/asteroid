import torch
from torch import testing
from asteroid.engine import Container
from asteroid.filterbanks import make_enc_dec
from asteroid.masknn import TDConvNet


def test_serialization():
    enc, dec = make_enc_dec('free', n_filters=512, kernel_size=16, stride=8)
    masker = TDConvNet(in_chan=512, n_src=2, out_chan=512)
    container = Container(enc, masker, dec)
    inp = torch.randn(2, 1, 16000)
    out = container(inp)
    # Serialize
    model_pack = container.serialize()
    # Load and forward
    new_model = Container(enc, masker, dec)
    new_model.load_model(model_pack)
    new_out = new_model(inp)
    # Check
    testing.assert_allclose(out, new_out)


def test_nocontainer():
    container = Container(None, None, None)
    inp = torch.randn(2, 512, 400)
    out = container(inp)
    testing.assert_allclose(inp, out)


def test_nomasker():
    enc, dec = make_enc_dec('free', n_filters=512, kernel_size=16, stride=8)
    container = Container(enc, None, dec)
    inp = torch.randn(2, 1, 16000)
    out = container(inp)
    assert inp.shape == out.shape
