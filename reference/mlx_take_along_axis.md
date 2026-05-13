# Take values using per-position axis indices

Mirrors
[`mlx.core.take_along_axis()`](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.take_along_axis.html)
while accepting 1-based R indices.

## Usage

``` r
mlx_take_along_axis(x, indices, axis)
```

## Arguments

- x:

  An mlx array.

- indices:

  Integer positions along `axis`. Must be broadcast-compatible with `x`
  except at the selected axis.

- axis:

  Axis to index (1-based).

## Value

An `mlx` array.

## Examples

``` r
x <- mlx_matrix(1:12, nrow = 3, ncol = 4)
idx <- matrix(c(1L, 4L,
                2L, 3L,
                4L, 1L), nrow = 3, byrow = TRUE)
mlx_take_along_axis(x, idx, axis = 2L)
#> mlx array [3 x 2]
#>   dtype: float32
#>   values:
#>      [,1] [,2]
#> [1,]    1   10
#> [2,]    5    8
#> [3,]   12    3
```
