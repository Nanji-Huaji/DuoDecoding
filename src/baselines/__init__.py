"""Compatibility imports for the baselines package.

Implementation globals live in ``src.baselines.core``. Patch that module when
replacing model/cache/simulator dependencies; exports here do not write through.
Legacy names are visible through imports/getattr/dir, not this package's __dict__.
"""

from . import core as _core
from .core import Baselines as Baselines

# Preserve the old module's star-import surface, including imported public names.
__all__ = [name for name in vars(_core) if not name.startswith("_")]


def __getattr__(name: str):
    """Keep historical explicit imports, including underscore helpers, working."""
    if not name.startswith("__") and name in vars(_core):
        return getattr(_core, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    legacy_names = {name for name in vars(_core) if not name.startswith("__")}
    return sorted(set(globals()) | legacy_names)
