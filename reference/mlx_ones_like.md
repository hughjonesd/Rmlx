# Ones shaped like an existing mlx array

`mlx_ones_like()` mirrors
[`mlx.core.ones_like()`](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.ones_like),
creating an array of ones with the same shape. Optionally override dtype
or device.

## Usage

``` r
mlx_ones_like(x, dtype = NULL, device = NULL)
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

An mlx array of ones matching `x`.

## See also

[mlx.core.ones_like](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.ones_like)

## Examples

``` r
base <- mlx_full(c(2, 3), 5)
ones <- mlx_ones_like(base)
as.matrix(ones)
#>      [,1] [,2] [,3]
#> [1,]    1    1    1
#> [2,]    1    1    1
```
