# Stop gradient propagation through an mlx array

Stop gradient propagation through an mlx array

## Usage

``` r
mlx_stop_gradient(x)
```

## Arguments

- x:

  An mlx array, or an R array/matrix/vector that will be converted via
  [`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md).

## Value

A new mlx array with identical values but zero gradient.

## See also

[mlx.core.stop_gradient](https://ml-explore.github.io/mlx/build/html/python/transforms.html#mlx.core.stop_gradient)

## Examples

``` r
x <- mlx_matrix(1:4, 2, 2)
mlx_stop_gradient(x)
#> mlx array [2 x 2]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1] [,2]
#> [1,]    1    3
#> [2,]    2    4
```
