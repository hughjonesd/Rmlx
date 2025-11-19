# Transpose of MLX matrix

Transpose of MLX matrix

## Usage

``` r
# S3 method for class 'mlx'
t(x)
```

## Arguments

- x:

  An mlx matrix (2-dimensional array).

## Value

The transposed MLX matrix.

## See also

[mlx.core.transpose](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.transpose)

## Examples

``` r
x <- as_mlx(matrix(1:6, 2, 3))
t(x)
#> mlx array [3 x 2]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1] [,2]
#> [1,]    1    2
#> [2,]    3    4
#> [3,]    5    6
```
