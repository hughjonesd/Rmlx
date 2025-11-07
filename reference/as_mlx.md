# Create MLX array from R object

Create MLX array from R object

## Usage

``` r
as_mlx(
  x,
  dtype = c("float32", "float64", "bool", "complex64", "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64"),
  device = mlx_default_device()
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

- device:

  Device: "gpu" (default) or "cpu"

## Value

An object of class `mlx`

## Details

### Default type behavior

When `dtype` is not specified:

- Numeric vectors/arrays (including R integers from `1:10`) → `float32`

- Logical vectors/arrays → `bool`

- Complex vectors/arrays → `complex64`

### Integer types require explicit dtype

**Important**: R integer vectors (like `1:10`) convert to `float32` by
default. To create integer MLX arrays, you must explicitly specify
`dtype`:

    x <- as_mlx(1:10, dtype = "int32")  # Creates int32 array
    x <- as_mlx(1:10)                    # Creates float32 array

This design avoids unintentional integer promotion, since R creates
integers in many contexts where floating-point is intended.

### Supported integer types

- **Signed**: `int8` (-128 to 127), `int16`, `int32`, `int64`

- **Unsigned**: `uint8` (0 to 255), `uint16`, `uint32`, `uint64`

### Type precision notes

- `float64` is supported but emits a warning and downcasts to `float32`

- Integer arithmetic may promote types (e.g., int32 + int32 might →
  int64)

- Mixed integer/float operations promote to float

### Missing values

MLX does not have an `NA` sentinel. When you pass numeric `NA` values
from R, they are stored as `NaN` inside MLX and returned to R as `NaN`.
Use [`is.nan()`](https://rdrr.io/r/base/is.finite.html) on MLX arrays
(method provided) if you need to detect them.

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
