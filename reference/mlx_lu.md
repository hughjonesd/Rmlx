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

  Execution target for APIs that expose a one-off device or stream
  override. Supply `"gpu"`, `"cpu"`, or an `mlx_stream` created via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  Ordinary array operations use the current
  [`mlx_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_device.md)
  instead.

## Value

A list with components `p` (pivot indices), `l` (lower triangular), and
`u` (upper triangular). The relationship is `A = L[P, ] %*% U`.

## Details

As of MLX 0.31.1, this operation only runs on CPU. Run it inside
[`with_device()`](https://hughjonesd.github.io/Rmlx/reference/with_device.md)
or
[`local_device()`](https://hughjonesd.github.io/Rmlx/reference/with_device.md),
or pass `device = "cpu"`.

## See also

[mlx.core.linalg.lu](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.core.linalg.lu)

## Examples

``` r
A <- mlx_matrix(rnorm(16), 4, 4)
lu_result <- mlx_lu(A, device = "cpu")
P <- lu_result$p  # Pivot indices
L <- lu_result$l  # Lower triangular
U <- lu_result$u  # Upper triangular
```
