# Cholesky decomposition for mlx arrays

If `x` is not symmetric positive semi-definite, "behaviour is undefined"
according to the MLX documentation.

## Usage

``` r
# S3 method for class 'mlx'
chol(x, pivot = FALSE, ...)
```

## Arguments

- x:

  An mlx matrix (2-dimensional array).

- pivot:

  Ignored; pivoted decomposition is not supported.

- ...:

  Additional arguments (unused).

## Value

Upper-triangular Cholesky factor as an mlx matrix.

## See also

[mlx.linalg.cholesky](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.cholesky)

## Examples

``` r
x <- as_mlx(matrix(c(4, 1, 1, 3), 2, 2))
chol(x)
#> mlx array [2 x 2]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1]     [,2]
#> [1,]    2 0.500000
#> [2,]    0 1.658312
```
