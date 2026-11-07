import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

# Tests share scripted fakes by module name; importlib collection does not add
# this directory itself (and the root test.py rules out a test.* package import).
test_str = str(Path(__file__).resolve().parent)
if test_str not in sys.path:
    sys.path.insert(0, test_str)
