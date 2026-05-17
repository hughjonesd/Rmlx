# Broadcast an array to a new shape

`mlx_broadcast_to()` mirrors
[`mlx.core.broadcast_to()`](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.broadcast_to),
repeating singleton dimensions without copying data.

## Usage

``` r
mlx_broadcast_to(x, shape)
```

## Arguments

- x:

  An mlx array.

- shape:

  Integer vector describing the broadcasted shape.

## Value

An mlx array with the requested dimensions. Dimnames from matching or
singleton broadcast axes are carried to the result.

## See also

[mlx.core.broadcast_to](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.broadcast_to)

## Examples

``` r
x <- mlx_matrix(1:3, nrow = 1)
broadcast <- mlx_broadcast_to(x, c(5, 3))
dim(broadcast)
#> [1] 5 3
```
