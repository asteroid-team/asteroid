import pytest

ignored_warnings = ["ignore:Could not log computational graph since"]

pytestmark = pytest.mark.filterwarnings(*ignored_warnings)
