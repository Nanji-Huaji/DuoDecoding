"""Compatibility imports for the baselines package.

Patch dependencies in their defining module: resource/policy assembly lives in
``src.baselines.runtime``, two-model adaptive in ``src.baselines.adaptive``,
CEE-SD in ``src.baselines.cee``, CU-HLM in ``src.baselines.cuhlm``, DSD/DSSD in
``src.baselines.dual``, local SD in ``src.dual_decoding``, shared primitives in
``src.baselines.operations`` and still-unmigrated baselines in
``src.baselines.core``; simulator factories use ``src.baselines.simulators``.
Exports here do not write through to an implementation module. Legacy names are
visible through imports/getattr/dir, not this package's __dict__.
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
