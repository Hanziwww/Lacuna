# Python Array API Standard – Quick Cheatsheet

> For libraries implementing the **Python Array API Standard** (e.g. `numpy.array_api`, compatible modes in CuPy, PyTorch, JAX, etc.).
> The goal: **write array-agnostic code** by only using this API.

---

## 0. Getting an array namespace

```py
# NumPy reference implementation
import numpy.array_api as xp

x = xp.asarray([1, 2, 3])

# From an existing array object (works for any conforming library)
xp = x.__array_namespace__()        # namespace object (the "xp" module-like thing)

# Introspect implementation
xp.__array_api_version__            # e.g. "2024.12"
info = xp.__array_namespace_info__()  # optional: capabilities, dtypes, devices, etc.
```

---

## 1. Core concepts

### Array object attributes

```py
x.dtype     # data type object
x.device    # where the data lives (CPU / GPU / etc., library-specific)
x.ndim      # number of dimensions
x.shape     # tuple of dimension sizes
x.size      # total number of elements
x.T         # transpose (last two axes swapped for >=2D)
x.mT        # matrix transpose (for 2D "matrix" semantics)
```

### Important methods

```py
xp = x.__array_namespace__()   # get namespace
y = x.to_device(device)        # move array to another device (if supported)
```

### Operators

All the usual Python operators are supported (obeying Array API semantics):

* Arithmetic: `+ - * / // % **`
* Matrix: `@` (same as `xp.matmul`)
* Bitwise: `& | ^ ~ << >>`
* Comparisons: `== != < <= > >=`

Broadcasting rules follow the standard’s broadcasting section (NumPy-like, but defined precisely in the spec).

---

## 2. Common keyword arguments (patterns)

Many functions share these keyword-only arguments:

* `dtype=` – result data type (if omitted, use default / promotion rules)
* `device=` – where to create the array
* `axis=` or `axis=(...)` – axis (or axes) to reduce / operate over
* `keepdims=` – keep reduced dimensions with size 1 (bool)
* `copy=` – whether to copy (for some functions like `astype`)

---

## 3. Constants

Available on the namespace:

```py
xp.e        # Euler’s number
xp.pi       # π
xp.inf      # positive infinity
xp.nan      # NaN
xp.newaxis  # same as None, used in indexing to add an axis
```

---

## 4. Creation functions

**Basic creation**

```py
xp.asarray(obj, /, *, dtype=None, device=None, copy=None)

xp.zeros(shape, *, dtype=None, device=None)
xp.ones(shape, *, dtype=None, device=None)
xp.full(shape, fill_value, *, dtype=None, device=None)
xp.empty(shape, *, dtype=None, device=None)

xp.zeros_like(x, *, dtype=None, device=None)
xp.ones_like(x, *, dtype=None, device=None)
xp.full_like(x, fill_value, *, dtype=None, device=None)
xp.empty_like(x, *, dtype=None, device=None)
```

**Ranges & sampling**

```py
xp.arange(start, stop=None, step=1, *, dtype=None, device=None)
xp.linspace(start, stop, num, *, dtype=None, device=None)
xp.meshgrid(*arrays, indexing='xy')   # grid of coordinates
```

**Matrices & triangles**

```py
xp.eye(n_rows, n_cols=None, k=0, *, dtype=None, device=None)  # identity-like
xp.tril(x, k=0)  # lower triangular
xp.triu(x, k=0)  # upper triangular
```

**Interop**

```py
xp.from_dlpack(dlpack_obj)  # create array from DLPack capsule
```

---

## 5. Dtype utilities

```py
y = xp.astype(x, dtype, /, *, copy=True)   # cast to new dtype

xp.can_cast(from_dtype, to_dtype, /)       # -> bool
xp.isdtype(dtype, kind)                    # kind: e.g. "real floating", "signed integer"
xp.result_type(*arrays_and_dtypes)         # -> promoted dtype

xp.finfo(dtype)  # floating-point info: eps, min, max, etc.
xp.iinfo(dtype)  # integer range info
```

---

## 6. Indexing

**Basic indexing**

```py
x[i]          # single index
x[i, j]
x[i:j]        # slice
x[i:j:k]      # slice with step
x[..., -1]    # ellipsis
x[:, xp.newaxis, :]  # add axis
```

**Boolean / integer array indexing**

```py
mask = (x > 0)
x_pos = x[mask]     # boolean mask

idx = xp.asarray([0, 2, 4])
x_sel = x[idx]      # integer indexing
```

**Indexing helper functions**

```py
xp.take(x, indices, /, *, axis=None)
xp.take_along_axis(x, indices, axis)
```

---

## 7. Shape & manipulation functions

```py
xp.reshape(x, newshape)                # view/reshape
xp.squeeze(x, axis=None)               # remove size-1 dimensions
xp.expand_dims(x, axis)                # insert new axis
xp.moveaxis(x, source, destination)
xp.permute_dims(x, axes)               # general permute

xp.stack(arrays, axis=0)               # stack along new axis
xp.concat(arrays, axis=0)              # concatenate along existing axis
xp.unstack(x, axis=0)                  # split into list along axis

xp.broadcast_to(x, shape)
xp.broadcast_arrays(*arrays)

xp.flip(x, axis=None)
xp.roll(x, shift, axis=None)
xp.repeat(x, repeats, axis=None)
xp.tile(x, reps)
```

---

## 8. Elementwise functions (per-element)

### 8.1 Arithmetic & comparisons

```py
xp.add(x, y)
xp.subtract(x, y)
xp.multiply(x, y)
xp.divide(x, y)
xp.floor_divide(x, y)
xp.remainder(x, y)
xp.pow(x, y)

xp.maximum(x, y)
xp.minimum(x, y)

xp.equal(x, y)
xp.not_equal(x, y)
xp.greater(x, y)
xp.greater_equal(x, y)
xp.less(x, y)
xp.less_equal(x, y)
```

### 8.2 Basic math

```py
xp.abs(x)
xp.negative(x)
xp.positive(x)
xp.sign(x)
xp.signbit(x)

xp.sqrt(x)
xp.square(x)

xp.floor(x)
xp.ceil(x)
xp.trunc(x)
xp.round(x)
```

### 8.3 Exponential & logarithmic

```py
xp.exp(x)
xp.expm1(x)       # exp(x) - 1, stable for small x
xp.log(x)
xp.log1p(x)       # log(1 + x)
xp.log2(x)
xp.log10(x)
xp.logaddexp(x, y)
```

### 8.4 Trigonometric & hyperbolic

```py
xp.sin(x)
xp.cos(x)
xp.tan(x)
xp.asin(x)
xp.acos(x)
xp.atan(x)
xp.atan2(y, x)

xp.sinh(x)
xp.cosh(x)
xp.tanh(x)
xp.asinh(x)
xp.acosh(x)
xp.atanh(x)
```

### 8.5 Logical & bitwise

```py
xp.logical_and(x, y)
xp.logical_or(x, y)
xp.logical_not(x)
xp.logical_xor(x, y)

xp.bitwise_and(x, y)
xp.bitwise_or(x, y)
xp.bitwise_xor(x, y)
xp.bitwise_invert(x)
xp.bitwise_left_shift(x, y)
xp.bitwise_right_shift(x, y)
```

### 8.6 NaN / inf utilities

```py
xp.isfinite(x)
xp.isinf(x)
xp.isnan(x)
xp.copysign(x, y)
xp.nextafter(x, y)
```

### 8.7 Complex helpers

```py
xp.real(x)    # real part
xp.imag(x)    # imaginary part
xp.conj(x)    # complex conjugate
```

---

## 9. Searching, sets & sorting

### 9.1 Searching

```py
xp.argmax(x, axis=None, keepdims=False)
xp.argmin(x, axis=None, keepdims=False)

xp.nonzero(x)            # indices of non-zero elements
xp.count_nonzero(x, axis=None, keepdims=False)

xp.where(cond, x, y)     # elementwise selection
xp.searchsorted(x, v, *, side='left')  # 1D sorted search
```

### 9.2 Set-like functions (1D)

All expect (and return) 1D arrays.

```py
xp.unique_values(x)      # sorted unique values
xp.unique_counts(x)      # (values, counts)
xp.unique_inverse(x)     # (values, inverse_indices)
xp.unique_all(x)         # all of the above in one call
```

### 9.3 Sorting

```py
xp.sort(x, axis=-1)
xp.argsort(x, axis=-1)
```

---

## 10. Reductions & statistics

All of these support `axis=` and `keepdims=`:

```py
xp.sum(x, axis=None, keepdims=False)
xp.prod(x, axis=None, keepdims=False)

xp.min(x, axis=None, keepdims=False)
xp.max(x, axis=None, keepdims=False)

xp.mean(x, axis=None, keepdims=False)
xp.var(x, axis=None, keepdims=False)
xp.std(x, axis=None, keepdims=False)

xp.cumulative_sum(x, axis=None)
xp.cumulative_prod(x, axis=None)
```

Boolean-style reductions (Utility functions):

```py
xp.all(x, axis=None, keepdims=False)
xp.any(x, axis=None, keepdims=False)

xp.diff(x, *, n=1, axis=-1)  # discrete difference along axis
```

---

## 11. Linear algebra

```py
# Matrix / tensor products
xp.matmul(x, y)                 # also used by the @ operator
xp.tensordot(x, y, *, axes=2)   # generalized tensor contraction
xp.vecdot(x, y, *, axis=None)   # dot along given axis (or last axis by default)

# Transpose helpers
xp.matrix_transpose(x)          # matrix-style transpose (last 2 axes)
x.T, x.mT                       # array / matrix transpose attributes
```

---

## 12. Devices & inspection

Use the **Inspection APIs** (via `__array_namespace_info__`) to write portable, device-aware code:

```py
info = xp.__array_namespace_info__()   # implementation-dependent mapping

info.devices()         # available devices (e.g. 'cpu', 'cuda:0', ...)
info.default_device()  # default device
info.dtypes()          # available dtypes
info.default_dtypes()  # default dtypes for kinds of input
info.capabilities()    # supported optional features
```

Exact structure of `info` is determined by the standard & implementation.

---

## 13. Broadcasting & type promotion (mental model)

* **Broadcasting**: follows NumPy-style compatible shapes (right-align shapes; each dimension is equal or 1 or one of them is 1 → expand) defined precisely in the spec.
* **Type promotion**:

  * Operations follow standard promotion tables for integers, floats, and mixing them.
  * Use `xp.result_type(...)` to check what dtype an operation would choose.
  * Mixing arrays with Python scalars is also defined in the spec; avoid relying on library-specific quirks.

Example:

```py
dtype = xp.result_type(x, y, 1.0)  # what dtype will x + y + 1.0 use?
z = xp.astype(x, dtype)
```

---

## 14. Interop & non-standard stuff

* **Random**, **FFT**, **sparse**, etc. are **not** part of the core Array API standard.
  If you need them, use the library’s own extension modules (e.g. `numpy.random`, `torch.random`) but be aware that this breaks strict array-agnostic portability.
* Use DLPack (`x.__dlpack__`, `xp.from_dlpack`) for zero-copy interoperability between different array libraries.

```py
capsule = x.__dlpack__()
y = xp.from_dlpack(capsule)
```

---

## 15. Minimal pattern for array-agnostic code

```py
def my_fn(x):
    # Get namespace from the input array
    xp = x.__array_namespace__()

    # Use only Array API standard functions
    x = xp.asarray(x)
    x = xp.astype(x, xp.float32)

    m = xp.mean(x, axis=0, keepdims=True)
    s = xp.std(x, axis=0, keepdims=True)

    z = (x - m) / s          # broadcasting + elementwise ops
    return z
```

If you can run your code with `numpy.array_api` and another Array API library without changes, you’re using the standard correctly.

---

**Reference:** This cheatsheet summarizes the official *Python Array API standard* (version 2024.12) categories and function names, including array object attributes, creation, elementwise, manipulation, searching, sorting, statistical, linear algebra, and utility functions as listed in the current specification. ([data-apis.org][1])

[1]: https://data-apis.org/array-api/latest/API_specification/ "API specification — Python array API standard 2024.12 documentation"
