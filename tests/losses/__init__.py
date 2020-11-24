import pytest

ignored_warnings = [
    "ignore:Could not log computational graph since",
    "ignore:The dataloader, val dataloader",
    "ignore:The dataloader, train dataloader",
]

pytestmark = pytest.mark.filterwarnings(*ignored_warnings)
