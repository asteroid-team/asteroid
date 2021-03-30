import torch
from asteroid.models.benchmarker import Benchmarker


def test_benchmark():
    model = nn.Identity()
    inputs_shape = (1, 100)
    bench = Benchmarker(model, inputs_shape, repeat=1)
    bench.compute_flops()
    bench.compute_memory_usage()
    bench.compute_inference_time()
