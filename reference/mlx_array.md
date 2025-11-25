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

  Numeric, logical, or complex vector. `data` is recycled to match
  dimensions according to R rules (but with an error if it doesn't tile
  into the dimensions exactly).

- dim:

  Integer vector of array dimensions. Set `dim = integer(0)` for a
  scalar, in which case `data` must be length 1.

- dtype:

  Optional MLX dtype. Defaults to `"float32"` for numeric input,
  `"bool"` for logical, and `"complex64"` for complex.

- device:

  Execution target: supply `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  Defaults to the current
  [`mlx_default_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_default_device.md)
  unless noted otherwise (helpers that act on an existing array
  typically reuse that array's device or stream).

## Value

An `mlx` array with the requested shape.

## Examples

``` r
payload <- runif(6)
mlx_array(payload, dim = c(2, 3))
#> mlx array [2 x 3]
#>   dtype: float32
#>   device: gpu
#>   values:
#>           [,1]      [,2]      [,3]
#> [1,] 0.9970691 0.5185567 0.7182697
#> [2,] 0.1490355 0.8461201 0.2413140
```
