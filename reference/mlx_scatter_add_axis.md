# Add values using per-position axis indices

Mirrors
[`mlx.core.scatter_add_axis()`](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.scatter_add_axis)
while accepting 1-based R indices.

## Usage

``` r
mlx_scatter_add_axis(x, indices, values, axis)
```

## Arguments

- x:

  An mlx array.

- indices:

  Integer positions along `axis`. Must be broadcast-compatible with `x`
  except at the selected axis.

- values:

  Replacement values.

- axis:

  Axis to index (1-based).

## Value

An updated `mlx` array after additive scatter.

## Examples

``` r
x <- mlx_matrix(1:12, nrow = 3, ncol = 4)
idx <- matrix(c(1L, 1L,
                2L, 3L,
                4L, 4L), nrow = 3, byrow = TRUE)
values <- matrix(c(10, 20,
                   30, 40,
                   50, 60), nrow = 3, byrow = TRUE)
mlx_scatter_add_axis(x, idx, values, axis = 2L)
#> mlx array [3 x 4]
#>   dtype: float32
#>   values:
#>      [,1] [,2] [,3] [,4]
#> [1,]   31    4    7   10
#> [2,]    2   35   48   11
#> [3,]    3    6    9  122
```
