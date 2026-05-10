# LU factorization

Computes the LU factorization of a matrix.

## Usage

``` r
mlx_lu(x, device = NULL)
```

## Arguments

- x:

  An mlx array.

- device:

  Execution target: supply `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  By default, many functions use the
  [`mlx_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_device.md)
  of their first argument.

## Value

A list with components `p` (pivot indices), `l` (lower triangular), and
`u` (upper triangular). The relationship is `A = L[P, ] %*% U`.

## Details

As of MLX 0.31.1, this operation only runs on CPU. Create or cast the
operands with `device = "cpu"` explicitly, or pass a `device = "cpu"`
argument. (Passing the argument won't affect the device of any mlx
object returned, just where this particular operation is run.)

## See also

[mlx.core.linalg.lu](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.core.linalg.lu)

## Examples

``` r
A <- mlx_matrix(rnorm(16), 4, 4, device = "cpu")
lu_result <- mlx_lu(A)
P <- lu_result$p  # Pivot indices
L <- lu_result$l  # Lower triangular
U <- lu_result$u  # Upper triangular
```
