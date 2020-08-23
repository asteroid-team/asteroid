import pytest
from torch import nn, optim
from asteroid.engine import optimizers
from torch_optimizer import Ranger


def optim_mapping():
    mapping_list = [
        (optim.Adam, "adam"),
        (optim.SGD, "sgd"),
        (optim.RMSprop, "rmsprop"),
        (Ranger, "ranger"),
    ]
    return mapping_list


global_model = nn.Sequential(nn.Linear(10, 10), nn.ReLU())


@pytest.mark.parametrize(
    "opt",
    [
        "Adam",
        "RMSprop",
        "SGD",
        "Adadelta",
        "Adagrad",
        "Adamax",
        "AdamW",
        "ASGD",
        "AccSGD",
        "AdaBound",
        "AdaMod",
        "DiffGrad",
        "Lamb",
        "NovoGrad",
        "PID",
        "QHAdam",
        "QHM",
        "RAdam",
        "SGDW",
        "Yogi",
        "Ranger",
        "RangerQH",
        "RangerVA",
    ],
)
def test_all_get(opt):
    optimizers.get(opt)(global_model.parameters(), lr=1e-3)


@pytest.mark.parametrize("opt_tuple", optim_mapping())
def test_get_str_returns_instance(opt_tuple):
    torch_optim = opt_tuple[0](global_model.parameters(), lr=1e-3)
    asteroid_optim = optimizers.get(opt_tuple[1])(global_model.parameters(), lr=1e-3)
    assert type(torch_optim) == type(asteroid_optim)
    assert torch_optim.param_groups == asteroid_optim.param_groups


@pytest.mark.parametrize("opt", [optim.Adam, optim.SGD, optim.Adadelta])
def test_get_instance_returns_instance(opt):
    torch_optim = opt(global_model.parameters(), lr=1e-3)
    asteroid_optim = optimizers.get(torch_optim)
    assert torch_optim == asteroid_optim


@pytest.mark.parametrize("wrong", ["wrong_string", 12, object()])
def test_get_errors(wrong):
    with pytest.raises(ValueError):
        # Should raise for anything not a Optimizer instance + unknown string
        optimizers.get(wrong)


def test_make_optimizer():
    optimizers.make_optimizer(global_model.parameters(), "adam", lr=1e-3)


def test_register():
    class Custom(optim.Optimizer):
        def __init__(self):
            super().__init__()

    optimizers.register_optimizer(Custom)
    cls = optimizers.get("Custom")
    assert cls == Custom

    with pytest.raises(ValueError):
        optimizers.register_optimizer(optimizers.Adam)
