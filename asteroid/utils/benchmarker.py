import torch.autograd.profiler as profiler
import torch
from pypapi import events, papi_high as high
import timeit
import numpy as np

from pypapi.exceptions import PapiNoEventError


class Benchmarker:
    """Wrapper around benchmark libraries on CPU. For inference time, memory
    usage and flops measurements.

    Args:
        model (torch.nn.Module): Instance of model.
        inputs_shape (list(int)): A sequence of integers defining the shape of the output tensor.
        repeat (int,optional) : The number of measurement to make for inference time measurement.

    Examples:
        Examples should be written in doctest format, and should illustrate how
        to use the function.
        >>> from asteroid.utils.benchmarker import Benchmarker
        >>> from asteroid.models import ConvTasNet

        >>> inputs_shape = [800]
        >>> model = ConvTasNet(n_src=1)
        >>> A = Benchmarker(model, inputs_shape, repeat=10)
        >>> print(A.compute_inference_time())
        Mean inference time 0.041 standard deviation 0.001 Min 0.040 in seconds
        >>> print(A.compute_memory_usage())
        -------------------------------  ------------  ------------ ...
                           Name    Self CPU %      Self CPU         ...
        -------------------------------  ------------  ------------ ...
                   aten::conv1d         0.01%       3.105us         ...
        >>>print(A.compute_flops())
        [11639695]
    """

    def __init__(self, model, inputs_shape, repeat=1000, decimal=3):

        self.model = model
        self.inputs_shape = inputs_shape
        self.repeat = repeat
        self.decimal = decimal
        # Assume [-1,1) inputs
        self.inputs = (torch.randn(self.inputs_shape) - 0.5) * 2
        # warm_up
        self.model(self.inputs)

    def process_rand_inputs(self):
        """Process inputs"""
        return self.model(self.inputs)

    def compute_inference_time(self):
        """Computes inference time 'repeat' number of times and prints staistics"""
        t0 = timeit.Timer(lambda: self.process_rand_inputs())
        T = t0.repeat(self.repeat, 1)
        return (
            f"Mean inference time {np.around(np.mean(T),decimals=self.decimal)}, standard deviation"
            f" {np.around(np.std(T),decimals=self.decimal)}, Min {np.around(np.min(T),decimals=self.decimal)} in seconds"
        )

    def compute_memory_usage(self):
        """Compute memory usage report"""
        with profiler.profile(profile_memory=True, record_shapes=True) as prof:
            with profiler.record_function("model_inference"):
                self.process_rand_inputs()
        return prof.table(sort_by="cpu_memory_usage")

    def compute_flops(self):
        """ Compute FLOPs """
        try:
            high.start_counters(
                [
                    events.PAPI_FP_OPS,
                ]
            )
        except PapiNoEventError:
            print(
                "Couldn't compute FLOPS. Try running sudo sh -c 'echo 1 >/proc/sys/kernel/perf_event_paranoid' "
                "if the problem persist, then your hardware might be incompatible "
            )
            return
        self.process_rand_inputs()
        x = high.stop_counters()
        return x
