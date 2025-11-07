# Compute matrix inverse via Cholesky decomposition

Computes the inverse of a positive definite matrix from its Cholesky
factor. Note: `x` should be the Cholesky factor (L or U), not the
original matrix.

## Usage

``` r
mlx_cholesky_inv(x, upper = FALSE)
```

## Arguments

- x:

  An mlx array.

- upper:

  Logical; if `TRUE`, `x` is upper triangular, otherwise lower
  triangular.

## Value

The inverse of the original matrix (A^-1 where A = LL' or A = U'U).

## Details

For a more R-like interface, see
[`chol2inv()`](https://hughjonesd.github.io/Rmlx/reference/chol2inv.md).

## See also

[`chol2inv()`](https://hughjonesd.github.io/Rmlx/reference/chol2inv.md),
[mlx.core.linalg.cholesky_inv](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.core.linalg.cholesky_inv)

## Examples

``` r
# Create a positive definite matrix
A <- matrix(rnorm(9), 3, 3)
A <- t(A) %*% A
# Compute Cholesky factor
L <- chol(A, pivot = FALSE, upper = FALSE)
# Get inverse from Cholesky factor
A_inv <- mlx_cholesky_inv(as_mlx(L))
```
