__array_api_version__ = "2024.12"

from .constants import e, inf, nan, newaxis, pi
from .creation import (
    arange,
    asarray,
    empty,
    empty_like,
    eye,
    from_dlpack,
    full,
    full_like,
    linspace,
    meshgrid,
    ones,
    ones_like,
    tril,
    triu,
    zeros,
    zeros_like,
)
from .dtypes import (
    astype,
    can_cast,
    finfo,
    iinfo,
    isdtype,
    result_type,
)
from .elementwise.arithmetic import (
    add,
    divide,
    equal,
    floor_divide,
    greater,
    greater_equal,
    less,
    less_equal,
    maximum,
    minimum,
    multiply,
    not_equal,
    pow,
    remainder,
    subtract,
)
from .elementwise.complex import conj, imag, real
from .elementwise.exp_log import (
    exp,
    expm1,
    log,
    log1p,
    log2,
    log10,
    logaddexp,
)
from .elementwise.logical_bitwise import (
    bitwise_and,
    bitwise_invert,
    bitwise_left_shift,
    bitwise_or,
    bitwise_right_shift,
    bitwise_xor,
    logical_and,
    logical_not,
    logical_or,
    logical_xor,
)
from .elementwise.math_basic import (
    abs,
    ceil,
    floor,
    negative,
    positive,
    round,
    sign,
    signbit,
    sqrt,
    square,
    trunc,
)
from .elementwise.nan_inf import copysign, isfinite, isinf, isnan, nextafter
from .elementwise.trig_hyp import (
    acos,
    acosh,
    asin,
    asinh,
    atan,
    atan2,
    atanh,
    cos,
    cosh,
    sin,
    sinh,
    tan,
    tanh,
)
from .indexing import take, take_along_axis
from .linalg import matmul, matrix_transpose, tensordot, vecdot
from .manipulation import (
    broadcast_arrays,
    broadcast_to,
    concat,
    expand_dims,
    flip,
    moveaxis,
    permute_dims,
    repeat,
    reshape,
    roll,
    squeeze,
    stack,
    tile,
    unstack,
)
from .reductions import (
    all,
    any,
    cumulative_prod,
    cumulative_sum,
    diff,
    max,
    mean,
    min,
    prod,
    std,
    sum,
    var,
)
from .searching import (
    argmax,
    argmin,
    count_nonzero,
    nonzero,
    searchsorted,
    where,
)
from .sets import unique_all, unique_counts, unique_inverse, unique_values
from .sorting import argsort, sort


def __getattr__(name: str):
    from . import _namespace as _ns

    return getattr(_ns, name)


def __array_namespace_info__():
    from . import _namespace as _ns

    return _ns.__array_namespace_info__()
