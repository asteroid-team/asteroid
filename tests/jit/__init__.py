import pytest

ignored_warnings = [
    "ignore:torch.tensor results are registered as constants in the trace.",
    "ignore:Converting a tensor to a Python boolean might cause the trace to be incorrect.",
    "ignore:Converting a tensor to a Python float might cause the trace to be incorrect.",
    "ignore:Using or importing the ABCs from",
]

pytestmark = pytest.mark.filterwarnings(*ignored_warnings)
