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
#> [1,] 0.7064338 0.1803388 0.6801629
#> [2,] 0.9485766 0.2168999 0.4988456
```
