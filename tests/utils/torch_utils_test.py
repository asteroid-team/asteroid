import torch
import pytest
from torch import nn
from asteroid import torch_utils


def test_pad():
    x = torch.randn(10, 1, 16000)
    y = torch.randn(10, 1, 16234)
    padded_x = torch_utils.pad_x_to_y(x, y)
    assert padded_x.shape == y.shape


def test_pad_fail():
    x = torch.randn(10, 16000, 1)
    y = torch.randn(10, 16234, 1)
    with pytest.raises(NotImplementedError):
        torch_utils.pad_x_to_y(x, y, axis=1)


def test_model_equal():
    model = nn.Sequential(nn.Linear(10, 10))
    assert torch_utils.are_models_equal(model, model)
    model_2 = nn.Sequential(nn.Linear(10, 10))
    assert not torch_utils.are_models_equal(model, model_2)


def test_loader_module():
    model = nn.Sequential(nn.Linear(10, 10))
    state_dict = model.state_dict()
    model_2 = nn.Sequential(nn.Linear(10, 10))
    model_2 = torch_utils.load_state_dict_in(state_dict, model_2)
    assert torch_utils.are_models_equal(model, model_2)


def test_loader_submodule():
    class SuperModule(nn.Module):
        """nn.Module subclass that holds a model under self.whoever"""

        def __init__(self, sub_model):
            super().__init__()
            self.whoever = sub_model

    model = SuperModule(nn.Sequential(nn.Linear(10, 10)))
    # Keys in state_dict will be whoever.0.weight, whoever.0.bias
    state_dict = model.state_dict()
    # We want to load it in model_2 (has keys 0.weight, 0.bias)
    model_2 = nn.Sequential(nn.Linear(10, 10))
    # Keys are not the same, torch raises an error for that.
    with pytest.raises(RuntimeError):
        model_2.load_state_dict(state_dict)
    # We can try loose model loading (assert it doesn't work)
    model_2.load_state_dict(state_dict, strict=False)
    assert not torch_utils.are_models_equal(model, model_2)
    # Apply our workaround torch_utils.load_state_dict_in and assert True.
    model_2 = torch_utils.load_state_dict_in(state_dict, model_2)
    assert torch_utils.are_models_equal(model, model_2)


@pytest.mark.parametrize(
    "data, expected",
    (
        (torch.tensor([1]), torch.tensor([1])),
        (torch.tensor([1, 2]), torch.tensor([2])),
        (torch.tensor([[1], [2]]), torch.tensor([2, 1])),
        (torch.tensor([[2, 5, 5], [3, 8, -2]]), torch.tensor([2, 3])),
    ),
)
def test_jitable_shape(data, expected):
    output = torch_utils.jitable_shape(data)
    assert torch.equal(output, expected)


def test_get_device():
    # We have only CPU in CI so can't test with real devices.

    class FakeTensor:
        device = "dev1"

    class FakeModule:
        def parameters(self):
            yield FakeTensor()

    class UnknownObject:
        pass

    assert torch_utils.get_device(FakeTensor()) == "dev1"
    assert torch_utils.get_device(FakeModule()) == "dev1"
    with pytest.raises(TypeError):
        torch_utils.get_device(UnknownObject())
