# Eigen decomposition for mlx arrays

Eigen decomposition for mlx arrays

## Usage

``` r
mlx_eig(x)
```

## Arguments

- x:

  An mlx matrix (2-dimensional array).

## Value

A list with components `values` and `vectors`, both mlx arrays.

## See also

[mlx.linalg.eig](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.eig)

## Examples

``` r
x <- as_mlx(matrix(c(2, -1, 0, 2), 2, 2))
eig <- mlx_eig(x)
eig$values
#> mlx array [2]
#>   dtype: complex64
#>   device: gpu
#>   values:
#> [1] 2+0i 2+0i
eig$vectors
#> mlx array [2 x 2]
#>   dtype: complex64
#>   device: gpu
#>   values:
#>                 [,1] [,2]
#> [1,] 2.384186e-07+0i 0+0i
#> [2,] 1.000000e+00+0i 1+0i
```
