# Write values using per-position axis indices

Mirrors
[`mlx.core.put_along_axis()`](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.put_along_axis.html)
while accepting 1-based R indices.

## Usage

``` r
mlx_put_along_axis(x, indices, values, axis)
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

An updated `mlx` array.

## Examples

``` r
x <- mlx_matrix(1:12, nrow = 3, ncol = 4)
idx <- matrix(c(1L, 4L,
                2L, 3L,
                4L, 1L), nrow = 3, byrow = TRUE)
values <- matrix(c(100, 200,
                   300, 400,
                   500, 600), nrow = 3, byrow = TRUE)
mlx_put_along_axis(x, idx, values, axis = 2L)
#> mlx array [3 x 4]
#>   dtype: float32
#>   device: cpu
#>   values:
#>      [,1] [,2] [,3] [,4]
#> [1,]  100    4    7  200
#> [2,]    2  300  400   11
#> [3,]  600    6    9  500
```
