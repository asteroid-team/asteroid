import torch

from asteroid.binarize import Binarize


def test_Binarize():
    # fmt: off
    inputs_list = [
        torch.Tensor([0.1, 0.6, 0.2, 0.6, 0.1, 0.1, 0.1, 0.7, 0.7, 0.7, 0.1, 0.7, 0.7, 0.7, 0.1,
                      0.8, 0.9, 0.2, 0.7, 0.1, 0.1, 0.1, 0.8, 0.1]),
        torch.Tensor([0.1, 0.1, 0.2, 0.1]),
        torch.Tensor([0.7, 0.7, 0.7, 0.7]),
        torch.Tensor([0.1, 0.7]),
    ]
    # fmt: on
    expected_result_list = [
        torch.Tensor([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]),
        torch.Tensor([0.0, 0.0, 0.0, 0.0]),
        torch.Tensor([1, 1, 1, 1]),
        torch.Tensor([0.0, 0.0]),
    ]
    binarizer = Binarize(0.5, 3, 1)
    for i in range(len(inputs_list)):
        result = binarizer(inputs_list[i].unsqueeze(0).unsqueeze(0))
        assert torch.allclose(result, expected_result_list[i])
