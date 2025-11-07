# Reshape an mlx array

Reshape an mlx array

## Usage

``` r
mlx_reshape(x, newshape)
```

## Arguments

- x:

  An mlx array.

- newshape:

  Integer vector specifying the new dimensions.

## Value

An mlx array with the specified shape.

## See also

[mlx.core.reshape](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.reshape.html)

## Examples

``` r
x <- as_mlx(1:12)
mlx_reshape(x, c(3, 4))
#> mlx array [3 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1] [,2] [,3] [,4]
#> [1,]    1    2    3    4
#> [2,]    5    6    7    8
#> [3,]    9   10   11   12
mlx_reshape(x, c(2, 6))
#> mlx array [2 x 6]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1] [,2] [,3] [,4] [,5] [,6]
#> [1,]    1    2    3    4    5    6
#> [2,]    7    8    9   10   11   12
```
