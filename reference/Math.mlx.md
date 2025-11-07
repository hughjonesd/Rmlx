# Math operations for MLX arrays

Math operations for MLX arrays

## Usage

``` r
# S3 method for class 'mlx'
Math(x, ...)
```

## Arguments

- x:

  An mlx array.

- ...:

  Additional arguments (ignored)

## Value

An mlx object with the result

## See also

[mlx.core.array](https://ml-explore.github.io/mlx/build/html/python/array.html)

## Examples

``` r
x <- as_mlx(matrix(c(-1, 0, 1), 3, 1))
sin(x)
#> mlx array [3 x 1]
#>   dtype: float32
#>   device: gpu
#>   values:
#>           [,1]
#> [1,] -0.841471
#> [2,]  0.000000
#> [3,]  0.841471
round(x + 0.4)
#> mlx array [3 x 1]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1]
#> [1,]   -1
#> [2,]    0
#> [3,]    1
```
