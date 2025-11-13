from ._runtime import set_num_threads, get_num_threads
try:
    from . import _core as _core
    print("DEBUG: imported _core from submodule")
except Exception as e:
    print("DEBUG: submodule import failed:", e)
    try:
        import _core as _core
        print("DEBUG: imported _core from top-level")
    except Exception as e2:
        print("DEBUG: top-level import also failed:", e2)
        _core = None
__version__ = getattr(_core, "version", "0.1.0")

__all__ = [
    "__version__",
    "set_num_threads",
    "get_num_threads",
]
