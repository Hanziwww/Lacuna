__array_api_version__ = "2022.12"

from .constants import e, pi, inf, nan, newaxis

from .creation import (
    asarray,
    zeros,
    ones,
    full,
    empty,
    zeros_like,
    ones_like,
    full_like,
    empty_like,
    arange,
    linspace,
    meshgrid,
    eye,
    tril,
    triu,
    from_dlpack,
)
from .dtypes import (
    astype,
    can_cast,
    isdtype,
    result_type,
    finfo,
    iinfo,
)
from .indexing import take, take_along_axis
from .manipulation import (
    reshape,
    squeeze,
    expand_dims,
    moveaxis,
    permute_dims,
    stack,
    concat,
    unstack,
    broadcast_to,
    broadcast_arrays,
    flip,
    roll,
    repeat,
    tile,
)
from .reductions import (
    sum,
    prod,
    min,
    max,
    mean,
    var,
    std,
    cumulative_sum,
    cumulative_prod,
    all,
    any,
    diff,
)
from .searching import (
    argmax,
    argmin,
    nonzero,
    count_nonzero,
    where,
    searchsorted,
)
from .sets import unique_values, unique_counts, unique_inverse, unique_all
from .sorting import sort, argsort
from .linalg import matmul, tensordot, vecdot, matrix_transpose
from .elementwise.arithmetic import (
    add,
    subtract,
    multiply,
    divide,
    floor_divide,
    remainder,
    pow,
    maximum,
    minimum,
    equal,
    not_equal,
    greater,
    greater_equal,
    less,
    less_equal,
)
from .elementwise.math_basic import (
    abs,
    negative,
    positive,
    sign,
    signbit,
    sqrt,
    square,
    floor,
    ceil,
    trunc,
    round,
)
from .elementwise.exp_log import (
    exp,
    expm1,
    log,
    log1p,
    log2,
    log10,
    logaddexp,
)
from .elementwise.trig_hyp import (
    sin,
    cos,
    tan,
    asin,
    acos,
    atan,
    atan2,
    sinh,
    cosh,
    tanh,
    asinh,
    acosh,
    atanh,
)
from .elementwise.logical_bitwise import (
    logical_and,
    logical_or,
    logical_not,
    logical_xor,
    bitwise_and,
    bitwise_or,
    bitwise_xor,
    bitwise_invert,
    bitwise_left_shift,
    bitwise_right_shift,
)
from .elementwise.nan_inf import isfinite, isinf, isnan, copysign, nextafter
from .elementwise.complex import real, imag, conj


def __getattr__(name: str):
    from . import _namespace as _ns
    return getattr(_ns, name)


def __array_namespace_info__():
    from . import _namespace as _ns
    return _ns.__array_namespace_info__()
