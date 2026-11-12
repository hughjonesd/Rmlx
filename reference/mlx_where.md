# Elementwise conditional selection

Elementwise conditional selection

## Usage

``` r
mlx_where(condition, x, y)
```

## Arguments

- condition:

  Logical mlx array (non-zero values are treated as `TRUE`).

- x:

  An mlx array, or an R array/matrix/vector that will be converted via
  [`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md).

- y:

  A second array, with the same accepted types as `x`.

## Value

An mlx array where elements are drawn from `x` when `condition` is
`TRUE`, otherwise from `y`.

## Details

Behaves like [`ifelse()`](https://rdrr.io/r/base/ifelse.html) for
arrays, but evaluates both branches. `condition`, `x`, and `y` must
broadcast to a common shape.

## See also

[mlx.core.where](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.where)

## Examples

``` r
cond <- mlx_matrix(c(TRUE, FALSE, TRUE, FALSE), 2, 2)
a <- mlx_matrix(1:4, 2, 2)
b <- mlx_matrix(5:8, 2, 2)
mlx_where(cond, a, b)
#> mlx array [2 x 2]
#>   dtype: float32
#>   values:
#>      [,1] [,2]
#> [1,]    1    3
#> [2,]    6    8
```
