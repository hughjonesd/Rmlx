# Transposed cross product

Transposed cross product

## Usage

``` r
# S3 method for class 'mlx'
tcrossprod(x, y = NULL, ...)
```

## Arguments

- x:

  An mlx matrix (2-dimensional array).

- y:

  An mlx matrix (default: NULL, uses x)

- ...:

  Additional arguments passed to base::tcrossprod.

## Value

`x %*% t(y)` as an mlx object.

## See also

[mlx.core.matmul](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.matmul)

## Examples

``` r
x <- mlx_matrix(1:6, 2, 3)
tcrossprod(x)
#> mlx array [2 x 2]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1] [,2]
#> [1,]   35   44
#> [2,]   44   56
```
