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
chol2inv(x, size = NCOL(x), ..., device = NULL)
```

## Arguments

- x:

  An mlx matrix (2-dimensional array).

- size:

  Ignored; included for compatibility with base R.

- ...:

  Additional arguments; ignored.

- device:

  Execution target: supply `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  By default, many functions use the
  [`mlx_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_device.md)
  of their first argument.

## Value

The inverse of the original matrix (before Cholesky decomposition).

## Details

As of MLX 0.31.1, this operation only runs on CPU. Create or cast the
operands with `device = "cpu"` explicitly, or pass a `device = "cpu"`
argument. (Passing the argument won't affect the device of any mlx
object returned, just where this particular operation is run.)

## See also

[`chol()`](https://rdrr.io/r/base/chol.html),
[`solve()`](https://rdrr.io/r/base/solve.html),
[`mlx_cholesky_inv()`](https://hughjonesd.github.io/Rmlx/reference/mlx_cholesky_inv.md)

## Examples

``` r
A <- mlx_matrix(c(4, 1, 1, 3), 2, 2, device = "cpu")
U <- chol(A)
A_inv <- chol2inv(U)
# Verify: A %*% A_inv should be identity
A %*% A_inv
#> mlx array [2 x 2]
#>   dtype: float32
#>   device: cpu
#>   values:
#>      [,1] [,2]
#> [1,]    1    0
#> [2,]    0    1
```
