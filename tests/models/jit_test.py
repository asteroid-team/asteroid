import torch
import pytest
from torch.testing import assert_allclose
from asteroid.models import DeMask
from torch.testing._internal.jit_utils import JitTestCase


class TestModels(JitTestCase):
    @staticmethod
    def _test_demask(self, device, check_export_import=True):

        example = torch.rand((1, 800), device=device)

        # set model to eval due to non-deterministic behaviour of dropout
        model = DeMask().eval().to(device)

        # test scripting of the separator
        torch.jit.script(model)

    def test_demask(self):
        self._test_demask(self, device='cpu')
