import torch
import pytest
from torch.testing import assert_allclose
from asteroid.models import DeMask, ConvTasNet, DPRNNTasNet, DPTNet, LSTMTasNet
from torch.testing._internal.jit_utils import JitTestCase


class TestModels(JitTestCase):
    @staticmethod
    def _test_demask(self, device, check_export_import=True):
        for fb in ["free", "stft", "analytic_free", "param_sinc"]:
            # set model to eval due to non-deterministic behaviour of dropout
            model = DeMask(fb_type=fb).eval().to(device)

            # test scripting of the separator
            torch.jit.script(model)

    def test_demask(self):
        self._test_demask(self, device="cpu")

    @staticmethod
    def _test_convtasnet(self, device, check_export_import=True):
        for fb in ["free", "stft", "analytic_free", "param_sinc"]:
            # set model to eval due to non-deterministic behaviour of dropout
            model = ConvTasNet(n_src=2, fb_name=fb).eval().to(device)

            # test scripting of the separator
            torch.jit.script(model)

    def test_convtasnet(self):
        self._test_convtasnet(self, device="cpu")


    @staticmethod
    def _test_drpnntasnet(self, device, check_export_import=True):
        for fb in ["free", "stft", "analytic_free", "param_sinc"]:
            # set model to eval due to non-deterministic behaviour of dropout
            model = DPRNNTasNet(n_src=2, fb_name=fb).eval().to(device)

            # test scripting of the separator
            torch.jit.script(model)

    def test_drpnntasnet(self):
        self._test_drpnntasnet(self, device="cpu")


    @staticmethod
    def _test_dptnet(self, device, check_export_import=True):
        for fb in ["free", "stft", "analytic_free", "param_sinc"]:
            # set model to eval due to non-deterministic behaviour of dropout
            model = DPTNet(n_src=2, fb_name=fb).eval().to(device)

            # test scripting of the separator
            torch.jit.script(model)

    def test_dptnet(self):
        self._test_dptnet(self, device="cpu")


    @staticmethod
    def _test_lstmtasnet(self, device, check_export_import=True):
        for fb in ["free", "stft", "analytic_free", "param_sinc"]:
            # set model to eval due to non-deterministic behaviour of dropout
            model = LSTMTasNet(n_src=2, fb_name=fb).eval().to(device)

            # test scripting of the separator
            torch.jit.script(model)

    def test_lstmtasnet(self):
        self._test_lstmtasnet(self, device="cpu")