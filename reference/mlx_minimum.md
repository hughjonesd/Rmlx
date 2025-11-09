# Elementwise minimum of two mlx arrays

Elementwise minimum of two mlx arrays

## Usage

``` r
mlx_minimum(x, y)
```

## Arguments

- x, y:

  mlx arrays or objects coercible with
  [`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md).

## Value

An mlx array containing the elementwise minimum.

## See also

[mlx.core.minimum](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.minimum)

## Examples

``` r
a <- as_mlx(matrix(1:4, 2, 2))
b <- as_mlx(matrix(c(4, 3, 2, 1), 2, 2))
mlx_minimum(a, b)
#> mlx array [2 x 2]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1] [,2]
#> [1,]    1    2
#> [2,]    2    1
```
