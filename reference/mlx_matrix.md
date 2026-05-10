# Construct MLX matrices efficiently

`mlx_matrix()` wraps
[`mlx_array()`](https://hughjonesd.github.io/Rmlx/reference/mlx_array.md)
for the common 2-D case. It accepts the same style arguments as
[`base::matrix()`](https://rdrr.io/r/base/matrix.html) but without
recycling, so mistakes surface early. Supply `nrow` or `ncol` (the other
may be inferred from `length(data)`).

## Usage

``` r
mlx_matrix(
  data,
  nrow = NULL,
  ncol = NULL,
  byrow = FALSE,
  dtype = NULL,
  device = mlx_default_device()
)
```

## Arguments

- data:

  Numeric, logical, or complex vector. `data` is recycled to match
  dimensions according to R rules (but with an error if it doesn't tile
  into the dimensions exactly).

- nrow, ncol:

  Matrix dimensions (positive integers).

- byrow:

  Logical; if `TRUE`, fill by rows (same semantics as
  [`base::matrix()`](https://rdrr.io/r/base/matrix.html)).

- dtype:

  Data type string. Supported types include:

  - Floating point: `"float32"`, `"float64"`

  - Integer: `"int8"`, `"int16"`, `"int32"`, `"int64"`, `"uint8"`,
    `"uint16"`, `"uint32"`, `"uint64"`

  - Other: `"bool"`, `"complex64"`

  `float64` arrays are CPU-only. Use `device = "cpu"` when creating or
  casting to `float64`, and cast back to `float32` before using the GPU.
  Not all functions support all types. See individual function
  documentation.

- device:

  Execution target: supply `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  By default, many functions use the
  [`mlx_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_device.md)
  of their first argument.

## Value

An `mlx` matrix with `dim = c(nrow, ncol)`.

## Examples

``` r
mlx_matrix(1:6, nrow = 2, ncol = 3, byrow = TRUE)
#> mlx array [2 x 3]
#>   dtype: float32
#>   device: cpu
#>   values:
#>      [,1] [,2] [,3]
#> [1,]    1    2    3
#> [2,]    4    5    6
```
