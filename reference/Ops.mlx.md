# Arithmetic and comparison operators for MLX arrays

Arithmetic and comparison operators for MLX arrays

## Usage

``` r
# S3 method for class 'mlx'
Ops(e1, e2 = NULL)
```

## Arguments

- e1:

  First operand (mlx or numeric)

- e2:

  Second operand (mlx or numeric)

## Value

An mlx object.

## See also

[mlx.core.array](https://ml-explore.github.io/mlx/build/html/python/array.html)

## Examples

``` r
x <- mlx_matrix(1:4, 2, 2)
y <- mlx_matrix(5:8, 2, 2)
x + y
#> mlx array [2 x 2]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1] [,2]
#> [1,]    6   10
#> [2,]    8   12
x < y
#> mlx array [2 x 2]
#>   dtype: bool
#>   device: gpu
#>   values:
#>      [,1] [,2]
#> [1,] TRUE TRUE
#> [2,] TRUE TRUE
```
