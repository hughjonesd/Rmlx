# Common parameters for MLX array creation

Print MLX array

## Usage

``` r
# S3 method for class 'mlx'
print(x, ...)
```

## Arguments

- x:

  An mlx array, or an R array/matrix/vector that will be converted via
  [`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md).

- ...:

  Additional arguments (ignored)

## Examples

``` r
x <- as_mlx(matrix(1:4, 2, 2))
print(x)
#> mlx array [2 x 2]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1] [,2]
#> [1,]    1    3
#> [2,]    2    4
```
