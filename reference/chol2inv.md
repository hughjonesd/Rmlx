# Inverse from Cholesky decomposition

Compute the inverse of a symmetric, positive definite matrix from its
Cholesky decomposition. The input `x` should be an upper triangular
matrix from [`chol()`](https://rdrr.io/r/base/chol.html).

## Usage

``` r
chol2inv(x, size = NCOL(x), ...)

# Default S3 method
chol2inv(x, size = NCOL(x), ...)

# S3 method for class 'mlx'
chol2inv(x, size = NCOL(x), ...)
```

## Arguments

- x:

  An mlx matrix (2-dimensional array).

- size:

  Ignored; included for compatibility with base R.

- ...:

  Additional arguments (unused).

## Value

The inverse of the original matrix (before Cholesky decomposition).

## See also

[`chol()`](https://rdrr.io/r/base/chol.html),
[`solve()`](https://rdrr.io/r/base/solve.html),
[`mlx_cholesky_inv()`](https://hughjonesd.github.io/Rmlx/reference/mlx_cholesky_inv.md)

## Examples

``` r
A <- mlx_matrix(c(4, 1, 1, 3), 2, 2)
U <- chol(A)
A_inv <- chol2inv(U)
# Verify: A %*% A_inv should be identity
A %*% A_inv
#> mlx array [2 x 2]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1] [,2]
#> [1,]    1    0
#> [2,]    0    1
```
