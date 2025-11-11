# Construct an MLX array from R data

`mlx_array()` is a low-level constructor that skips
[`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md)'s
type inference and dimension guessing. Supply the raw payload vector
plus an explicit shape and it pipes the data straight into MLX.

## Usage

``` r
mlx_array(
  data,
  dim,
  dtype = NULL,
  device = mlx_default_device(),
  allow_scalar = FALSE
)
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

- allow_scalar:

  Logical; set `TRUE` to permit `dim = integer(0)` so scalar payloads
  can be represented. When enabled, `data` must be length 1 and the
  resulting array is dimensionless.

## Value

An `mlx` array with the requested shape.

## Examples

``` r
payload <- runif(6)
arr <- mlx_array(payload, dim = c(2, 3))
as.matrix(arr)
#>           [,1]      [,2]       [,3]
#> [1,] 0.7182697 0.5470434 0.02795603
#> [2,] 0.2413140 0.8348018 0.46938431
```
