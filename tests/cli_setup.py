import torch
import soundfile as sf
import numpy as np
import os
from asteroid.models import ConvTasNet, save_publishable
from asteroid.data.wham_dataset import wham_noise_license, wsj0_license


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


def setup_upload():
    train_set_infos = dict(
        dataset="WHAM", task="sep_noisy", licenses=[wsj0_license, wham_noise_license]
    )
    final_results = {"si_sdr": 8.67, "si_sdr_imp": 13.16}
    model = ConvTasNet(
        n_src=2,
        n_repeats=2,
        n_blocks=3,
        bn_chan=16,
        hid_chan=4,
        skip_chan=8,
        n_filters=32,
    )
    model_dict = model.serialize()
    model_dict.update(train_set_infos)

    os.makedirs("publish_dir", exist_ok=True)
    save_publishable(
        "publish_dir",
        model_dict,
        metrics=final_results,
        train_conf=dict(),
    )


if __name__ == "__main__":
    setup_register_sr()
    setup_infer()
    setup_upload()
