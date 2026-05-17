# Broadcast multiple arrays to a shared shape

`mlx_broadcast_arrays()` mirrors
[`mlx.core.broadcast_arrays()`](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.broadcast_arrays),
returning a list of inputs expanded to a common shape.

## Usage

``` r
mlx_broadcast_arrays(...)
```

## Arguments

- ...:

  One or more arrays (or a single list) convertible via
  [`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md).

## Value

A list of broadcast mlx arrays, with each input's dimnames broadcast to
the shared shape where possible.

## See also

[mlx.core.broadcast_arrays](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.broadcast_arrays)

## Examples

``` r
a <- mlx_matrix(1:3, nrow = 1)
b <- mlx_matrix(1:3, ncol = 1)
outs <- mlx_broadcast_arrays(a, b)
lapply(outs, dim)
#> [[1]]
#> [1] 3 3
#> 
#> [[2]]
#> [1] 3 3
#> 
```
