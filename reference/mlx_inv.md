# Compute matrix inverse

Computes the inverse of a square matrix. Note that as of MLX 0.30.0,
this runs on the CPU.

## Usage

``` r
mlx_inv(x, device = NULL)
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

The inverse of `x`.

## Details

As of MLX 0.31.1, this operation only runs on CPU. Run it inside
[`with_device()`](https://hughjonesd.github.io/Rmlx/reference/with_device.md)
or
[`local_device()`](https://hughjonesd.github.io/Rmlx/reference/with_device.md),
or pass `device = "cpu"`.

## See also

[mlx.core.linalg.inv](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.core.linalg.inv)

## Examples

``` r
A <- mlx_matrix(c(4, 7, 2, 6), 2, 2)
A_inv <- mlx_inv(A, device = "cpu")
# Verify: A %*% A_inv should be identity
A %*% A_inv
#> mlx array [2 x 2]
#>   dtype: float32
#>   values:
#>              [,1] [,2]
#> [1,] 1.000000e+00    0
#> [2,] 4.768372e-07    1
```
