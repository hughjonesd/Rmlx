# Compute matrix inverse via Cholesky decomposition

Computes the inverse of a positive definite matrix from its Cholesky
factor. Note: `x` should be the Cholesky factor (L or U), not the
original matrix.

## Usage

``` r
mlx_cholesky_inv(x, upper = FALSE, device = NULL)
```

## Arguments

- x:

  An mlx array.

- upper:

  Logical; if `TRUE`, `x` is upper triangular, otherwise lower
  triangular.

- device:

  Execution target for APIs that expose a one-off device or stream
  override. Supply `"gpu"`, `"cpu"`, or an `mlx_stream` created via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  Ordinary array operations use the current
  [`mlx_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_device.md)
  instead.

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
mlx_cholesky_inv(as_mlx(L), device = "cpu")
#> mlx array [3 x 3]
#>   dtype: float32
#>   values:
#>           [,1]     [,2]     [,3]
#> [1,] 0.1525249 0.000000 0.000000
#> [2,] 0.0000000 1.428797 0.000000
#> [3,] 0.0000000 0.000000 2.136137
```
