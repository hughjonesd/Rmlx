# Zeros shaped like an existing mlx array

`mlx_zeros_like()` mirrors
[`mlx.core.zeros_like()`](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.zeros_like):
it creates a zero-filled array matching the source array's shape.
Optionally override the dtype or device.

## Usage

``` r
mlx_zeros_like(x, dtype = NULL, device = NULL)
```

## Arguments

- x:

  An mlx array.

- dtype:

  Optional MLX dtype override. Defaults to the source array's dtype.

- device:

  Execution target: supply `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  Defaults to the current
  [`mlx_default_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_default_device.md)
  unless noted otherwise (helpers that act on an existing array
  typically reuse that array's device or stream).

## Value

An mlx array of zeros matching `x`.

## See also

[mlx.core.zeros_like](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.zeros_like)

## Examples

``` r
base <- mlx_ones(c(2, 2))
zeros <- mlx_zeros_like(base)
as.matrix(zeros)
#>      [,1] [,2]
#> [1,]    0    0
#> [2,]    0    0
```
