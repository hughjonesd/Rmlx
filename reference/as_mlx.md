# Create MLX array from R object

Create MLX array from R object

## Usage

``` r
as_mlx(
  x,
  dtype = c("float32", "float64", "bool", "complex64", "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64")
)
```

## Arguments

- x:

  Numeric, logical, or complex vector, matrix, or array to convert

- dtype:

  Data type for the MLX array. One of:

  - Floating point: `"float32"`, `"float64"`

  - Integer signed: `"int8"`, `"int16"`, `"int32"`, `"int64"`

  - Integer unsigned: `"uint8"`, `"uint16"`, `"uint32"`, `"uint64"`

  - Other: `"bool"`, `"complex64"`

  If not specified, defaults to `"float32"` for numeric, `"bool"` for
  logical, and `"complex64"` for complex inputs.

## Value

An object of class `mlx`

### Integer types require explicit dtype

R integer vectors (like `1:10`) convert to `float32` by default. To
create integer MLX arrays, you must explicitly specify `dtype`:

    x <- as_mlx(1:10, dtype = "int32")  # Creates int32 array
    x <- as_mlx(1:10)                    # Creates float32 array

### Type precision

- `float64` is supported on CPU only. Use
  [`with_device()`](https://hughjonesd.github.io/Rmlx/reference/with_device.md)
  or
  [`local_device()`](https://hughjonesd.github.io/Rmlx/reference/with_device.md)
  to run float64 work on CPU.

- Integer arithmetic may promote types (e.g., int32 + int32 might →
  int64)

- Mixed integer/float operations promote to float

### Missing values

MLX does not have an `NA` sentinel. When you pass numeric `NA` values
from R, they are stored as `NaN` inside MLX and returned to R as `NaN`.
Use [`is.nan()`](https://rdrr.io/r/base/is.finite.html) on MLX arrays if
you need to detect them. [`is.na()`](https://rdrr.io/r/base/NA.html) on
mlx objects calls [`is.nan()`](https://rdrr.io/r/base/is.finite.html).

### Scalars

MLX allows scalar values, with a zero-length dimension (`integer(0)`).
These are not usually what R users want. `as_mlx()` never returns a
scalar; call `[mlx_reshape(x, integer(0))][mlx_reshape()]` to create one
explicitly, or use `[mlx_array(..., allow_scalar = TRUE)][mlx_array()]`.

## Details

MLX does not support `float64` operations on GPU. When this function
creates a `float64` array or converts one back to R, Rmlx temporarily
switches only that internal creation or layout work to CPU. Later
operations on the returned array still use the current
[`mlx_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_device.md).

## See also

[mlx.core.array](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.array)

[mlx-methods](https://hughjonesd.github.io/Rmlx/reference/mlx-methods.md)

## Examples

``` r
# Default float32 for numeric
x <- as_mlx(c(1.5, 2.5, 3.5))
mlx_dtype(x)  # "float32"
#> [1] "float32"

# R integers also default to float32
x <- as_mlx(1:10)
mlx_dtype(x)  # "float32"
#> [1] "float32"

# Explicit integer types
x_int <- as_mlx(1:10, dtype = "int32")
mlx_dtype(x_int)  # "int32"
#> [1] "int32"

# Unsigned integers
x_uint <- as_mlx(c(0, 128, 255), dtype = "uint8")

# Logical → bool
mask <- as_mlx(c(TRUE, FALSE, TRUE))
mlx_dtype(mask)  # "bool"
#> [1] "bool"
```
