from ._runtime import get_num_threads, set_num_threads

try:
    from . import _core as _core
except Exception as e:
    try:
        import _core as _core
    except Exception as e2:
        _core = None
__version__ = getattr(_core, "version", "0.1.0")

__all__ = [
    "__version__",
    "set_num_threads",
    "get_num_threads",
]
