# Eigen decomposition for mlx arrays

Eigen decomposition for mlx arrays

## Usage

``` r
mlx_eig(x, device = NULL)
```

## Arguments

- x:

  An mlx matrix (2-dimensional array).

- device:

  Execution target: supply `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  By default, many functions use the
  [`mlx_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_device.md)
  of their first argument.

## Value

A list with components `values` and `vectors`, both mlx arrays.

## Details

As of MLX 0.31.1, this operation only runs on CPU. Create or cast the
operands with `device = "cpu"` explicitly, or pass a `device = "cpu"`
argument. (Passing the argument won't affect the device of any mlx
object returned, just where this particular operation is run.)

## See also

[mlx.linalg.eig](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.eig)

## Examples

``` r
x <- mlx_matrix(c(2, -1, 0, 2), 2, 2, device = "cpu")
eig <- mlx_eig(x)
eig$values
#> mlx array [2]
#>   dtype: complex64
#>   device: cpu
#>   values:
#> [1] 2+0i 2+0i
eig$vectors
#> mlx array [2 x 2]
#>   dtype: complex64
#>   device: cpu
#>   values:
#>                 [,1] [,2]
#> [1,] 2.384186e-07+0i 0+0i
#> [2,] 1.000000e+00+0i 1+0i
```
