# Broadcast multiple arrays to a shared shape

`mlx_broadcast_arrays()` mirrors
[`mlx.core.broadcast_arrays()`](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.broadcast_arrays),
returning a list of inputs expanded to a common shape.

## Usage

``` r
mlx_broadcast_arrays(..., device = NULL)
```

## Arguments

- ...:

  One or more arrays (or a single list) convertible via
  [`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md).

- device:

  Execution target: supply `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  Default:
  [`mlx_default_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_default_device.md).

## Value

A list of broadcast mlx arrays.

## See also

[mlx.core.broadcast_arrays](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.broadcast_arrays)

## Examples

``` r
a <- as_mlx(matrix(1:3, nrow = 1))
b <- as_mlx(matrix(1:3, ncol = 1))
outs <- mlx_broadcast_arrays(a, b)
lapply(outs, dim)
#> [[1]]
#> [1] 3 3
#> 
#> [[2]]
#> [1] 3 3
#> 
```
