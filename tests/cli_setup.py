import torch
import soundfile as sf
import numpy as np
from asteroid.models import ConvTasNet


def setup_register_sr():
    model = ConvTasNet(
        n_src=2,
        n_repeats=2,
        n_blocks=3,
        bn_chan=16,
        hid_chan=4,
        skip_chan=8,
        n_filters=32,
    )
    to_save = model.serialize()
    to_save["model_args"].pop("sample_rate")
    torch.save(to_save, "tmp.th")


def setup_infer():
    sf.write("tmp.wav", np.random.randn(16000), 8000)
    sf.write("tmp2.wav", np.random.randn(16000), 8000)


if __name__ == "__main__":
    setup_register_sr()
    setup_infer()
