import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

# Tests share scripted fakes by module name; importlib collection does not add
# this directory itself (and the root test.py rules out a test.* package import).
test_str = str(Path(__file__).resolve().parent)
if test_str not in sys.path:
    sys.path.insert(0, test_str)


def pytest_addoption(parser):
    parser.addoption(
        "--require-golden-assets",
        action="store_true",
        default=False,
        help="缺少历史 golden checkpoint 时失败，而不是跳过（严格资产验收）。",
    )


@pytest.fixture
def require_golden_assets(request) -> bool:
    return bool(request.config.getoption("--require-golden-assets"))
