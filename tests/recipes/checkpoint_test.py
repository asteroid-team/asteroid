import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import yaml
from pytorch_lightning import Trainer


@pytest.mark.parametrize("resume", [None, "resume.ckpt"])
def test_variable_speakers_checkpoint(tmp_path, monkeypatch, resume):
    recipe = Path(__file__).resolve().parents[2] / "egs/wsj0-mix-var/Multi-Decoder-DPRNN"
    monkeypatch.syspath_prepend(str(recipe))
    with patch.dict(sys.modules):
        for module in ("model", "wsj0_mix_variable"):
            sys.modules.pop(module, None)
        main = runpy.run_path(str(recipe / "train.py"), init_globals={"yaml": yaml})["main"]
    dataset = torch.utils.data.TensorDataset(torch.zeros(1, 1))

    def make_model(conf, sample_rate):
        model = torch.nn.Linear(1, 1)
        return model, torch.optim.SGD(model.parameters(), lr=0.1)

    monkeypatch.setitem(main.__globals__, "Wsj0mixVariable", lambda **kwargs: dataset)
    monkeypatch.setitem(main.__globals__, "make_model_and_optimizer", make_model)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    checkpoint = str(tmp_path / resume) if resume else None
    conf = {
        "data": {
            "train_dir": "unused{}",
            "valid_dir": "unused{}",
            "sample_rate": 8000,
            "seglen": 1,
            "minlen": 1,
        },
        "masknet": {"n_srcs": [2]},
        "training": {
            "batch_size": 1,
            "num_workers": 0,
            "half_lr": False,
            "lr_decay": False,
            "epochs": 1,
            "early_stop": False,
        },
        "main_args": {"exp_dir": str(tmp_path), "resume_from": checkpoint},
        "loss": {"lambda": 0.05},
    }

    class StopBeforeTraining(Exception):
        pass

    # Keep the real constructor and fit signature, but do not start training.
    with patch.object(Trainer, "fit", autospec=True, side_effect=StopBeforeTraining) as fit:
        with pytest.raises(StopBeforeTraining):
            main(conf)
    assert fit.call_args.kwargs == {"ckpt_path": checkpoint}
    assert isinstance(fit.call_args.args[0], Trainer)
