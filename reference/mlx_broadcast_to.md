# Broadcast an array to a new shape

`mlx_broadcast_to()` mirrors
[`mlx.core.broadcast_to()`](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.broadcast_to),
repeating singleton dimensions without copying data.

## Usage

``` r
mlx_broadcast_to(x, shape, device = NULL)
```

## Arguments

- x:

  An mlx array.

- shape:

  Integer vector describing the broadcasted shape.

- device:

  Execution target: supply `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  Defaults to the current
  [`mlx_default_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_default_device.md)
  unless noted otherwise (helpers that act on an existing array
  typically reuse that array's device or stream).

## Value

An mlx array with the requested dimensions.

## See also

[mlx.core.broadcast_to](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.broadcast_to)

## Examples

``` r
x <- mlx_matrix(1:3, nrow = 1)
broadcast <- mlx_broadcast_to(x, c(5, 3))
dim(broadcast)
#> [1] 5 3
```
