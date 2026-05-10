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

  Execution target: supply `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  By default, many functions use the
  [`mlx_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_device.md)
  of their first argument.

## Value

The inverse of `x`.

## Details

As of MLX 0.31.1, this operation only runs on CPU. Create or cast the
operands with `device = "cpu"` explicitly, or pass a `device = "cpu"`
argument. (Passing the argument won't affect the device of any mlx
object returned, just where this particular operation is run.)

## See also

[mlx.core.linalg.inv](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.core.linalg.inv)

## Examples

``` r
A <- mlx_matrix(c(4, 7, 2, 6), 2, 2, device = "cpu")
A_inv <- mlx_inv(A)
# Verify: A %*% A_inv should be identity
A %*% A_inv
#> mlx array [2 x 2]
#>   dtype: float32
#>   device: cpu
#>   values:
#>              [,1] [,2]
#> [1,] 1.000000e+00    0
#> [2,] 4.768372e-07    1
```
