from ._runtime import set_num_threads, get_num_threads
try:
    from . import _core as _core
    __version__ = getattr(_core, "version", "0.1.0")
except Exception:  # pragma: no cover
    __version__ = "0.1.0"

__all__ = [
    "__version__",
    "set_num_threads",
    "get_num_threads",
]
