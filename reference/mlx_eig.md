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

  Execution target for APIs that expose a one-off device or stream
  override. Supply `"gpu"`, `"cpu"`, or an `mlx_stream` created via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  Ordinary array operations use the current
  [`mlx_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_device.md)
  instead.

## Value

A list with components `values` and `vectors`, both mlx arrays.

## Details

As of MLX 0.31.1, this operation only runs on CPU. Run it inside
[`with_device()`](https://hughjonesd.github.io/Rmlx/reference/with_device.md)
or
[`local_device()`](https://hughjonesd.github.io/Rmlx/reference/with_device.md),
or pass `device = "cpu"`.

## See also

[mlx.linalg.eig](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.eig)

## Examples

``` r
x <- mlx_matrix(c(2, -1, 0, 2), 2, 2)
eig <- mlx_eig(x, device = "cpu")
eig$values
#> mlx array [2]
#>   dtype: complex64
#>   values:
#> [1] 2+0i 2+0i
eig$vectors
#> mlx array [2 x 2]
#>   dtype: complex64
#>   values:
#>                 [,1] [,2]
#> [1,] 2.384186e-07+0i 0+0i
#> [2,] 1.000000e+00+0i 1+0i
```
