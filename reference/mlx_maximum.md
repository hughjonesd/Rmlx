# Elementwise maximum of two mlx arrays

Elementwise maximum of two mlx arrays

## Usage

``` r
mlx_maximum(x, y)
```

## Arguments

- x, y:

  mlx arrays or objects coercible with
  [`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md).

## Value

An mlx array containing the elementwise maximum.

## See also

[mlx.core.maximum](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.maximum)

## Examples

``` r
mlx_maximum(1:3, c(3, 2, 1))
#> mlx array []
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 3 2 3
```
