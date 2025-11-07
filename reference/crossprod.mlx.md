# Cross product

Cross product

## Usage

``` r
# S3 method for class 'mlx'
crossprod(x, y = NULL, ...)
```

## Arguments

- x:

  An mlx matrix (2-dimensional array).

- y:

  An mlx matrix (default: NULL, uses x)

- ...:

  Additional arguments passed to base::crossprod.

## Value

`t(x) %*% y` as an mlx object

## See also

[mlx.core.matmul](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.matmul)

## Examples

``` r
x <- as_mlx(matrix(1:6, 2, 3))
crossprod(x)
#> mlx array [3 x 3]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1] [,2] [,3]
#> [1,]    5   11   17
#> [2,]   11   25   39
#> [3,]   17   39   61
```
