__array_api_version__ = "2024.12"

from .constants import e, pi, inf, nan, newaxis


def __getattr__(name: str):
    from . import _namespace as _ns
    return getattr(_ns, name)


def __array_namespace_info__():
    from . import _namespace as _ns
    return _ns.__array_namespace_info__()
