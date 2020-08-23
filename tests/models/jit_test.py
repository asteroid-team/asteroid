import torch
import pytest
from torch.testing import assert_allclose
from asteroid.models import ConvTasNet, DPRNNTasNet, DPTNet, LSTMTasNet
from asteroid.models import SuDORMRFNet, SuDORMRFImprovedNet
from torch.testing._internal.jit_utils import JitTestCase


class TestModels(JitTestCase):
    @staticmethod
    def _test_drpnn(self, device, check_export_import=True):

        example = torch.rand((1, 800), device=device)

        # set model to eval due to non-deterministic behaviour of dropout
        model = DPRNNTasNet(
            n_src=2,
            n_repeats=2,
            n_blocks=2,
            bn_chan=16,
            hid_chan=4,
            skip_chan=8,
            n_filters=32,
            fb_name='stft',
        ).eval().to(device)

        # test trace
        # self.checkTrace(model, (example,), export_import=check_export_import)

        # test scripting of the separator
        torch.jit.script(model)


    def test_drpnn(self):
        self._test_drpnn(self, device='cpu')
