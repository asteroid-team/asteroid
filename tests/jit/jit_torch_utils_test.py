import pytest
import torch

import asteroid.utils.torch_utils as torch_utils


@pytest.mark.parametrize(
    "data",
    (
        torch.tensor([1]),
        torch.tensor([1, 2]),
        torch.tensor([[1], [2]]),
        torch.tensor([[2, 5], [3, 8]]),
    ),
)
def test_get_shape(data):
    expected = torch_utils.get_shape(data)
    scripted = torch.jit.trace(torch_utils.get_shape, torch.tensor([1]))
    output = scripted(data)
    assert torch.equal(output, expected)
