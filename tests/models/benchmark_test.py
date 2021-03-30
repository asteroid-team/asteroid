import torch
from asteroid.models.benchmarker import Benchmarker


class Dummy_model(torch.nn.Module):
    def __init__(self):
        super(Dummy_model, self).__init__()
        self.layer = torch.nn.Identity()

    def forward(self, x):
        return self.layer(x)


def test_benchmark():
    model = Dummy_model()
    inputs_shape = (1, 100)
    bench = Benchmarker(model, inputs_shape, repeat=1)
    bench.compute_flops()
    bench.compute_memory_usage()
    bench.compute_inference_time()
