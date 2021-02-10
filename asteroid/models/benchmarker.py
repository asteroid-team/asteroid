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
        >>> from asteroid.models.benchmarker import Benchmarker
        >>> from asteroid.models import ConvTasNet

        >>> inputs_shape = [800]
        >>> model = ConvTasNet(n_src=1)
        >>> A = Benchmarker(model, inputs_shape, repeat=10)
        >>> print(A.compute_inference_time())
        Mean inference time 0.04166898940093233 standard deviation 0.0014542574214454578 Min 0.04037960300047416 in seconds
        >>> print(A.compute_memory_usage())
        -------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ---------------------------------------------
                           Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls                                   Input Shapes
        -------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ---------------------------------------------
                   aten::conv1d         0.01%       3.105us         0.55%     283.483us     283.483us     247.50 Kb           0 b             1  [[1, 128, 99], [512, 128, 1], [512], [], [],
        >>>print(A.compute_flops())
        [11639695]
    """

    def __init__(self, model, inputs_shape, repeat=1000):

        self.model = model
        self.inputs_shape = inputs_shape
        self.repeat = repeat
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
            f"Mean inference time {np.mean(T)} standard deviation"
            f" {np.std(T)} Min {np.min(T)} in seconds"
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
