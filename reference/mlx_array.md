# Construct an MLX array from R data

`mlx_array()` is a low-level constructor that skips
[`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md)'s
type inference and dimension guessing. Supply the raw payload vector
plus an explicit shape and it pipes the data straight into MLX.

## Usage

``` r
mlx_array(data, dim, dtype = NULL, device = mlx_default_device())
```

## Arguments

- data:

  Numeric, logical, or complex vector supplying the payload. Any
  dimension attributes are ignored; pass `dim` explicitly.

- dim:

  Integer vector of array dimensions (product must equal
  `length(data)`).

- dtype:

  Optional MLX dtype. Defaults to `"float32"` for numeric input,
  `"bool"` for logical, and `"complex64"` for complex.

- device:

  Execution target: provide `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  Defaults to the current
  [`mlx_default_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_default_device.md).

## Value

An `mlx` array with the requested shape.

## Examples

``` r
payload <- runif(6)
arr <- mlx_array(payload, dim = c(2, 3))
as.matrix(arr)
#>           [,1]      [,2]       [,3]
#> [1,] 0.4936370 0.2041783 0.06521611
#> [2,] 0.7793086 0.7133973 0.35420680
```
