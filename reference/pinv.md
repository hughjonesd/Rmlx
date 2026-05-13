# Moore-Penrose pseudoinverse for MLX arrays

Moore-Penrose pseudoinverse for MLX arrays

## Usage

``` r
pinv(x, device = NULL)
```

## Arguments

- x:

  An mlx object or coercible matrix.

- device:

  Execution target for APIs that expose a one-off device or stream
  override. Supply `"gpu"`, `"cpu"`, or an `mlx_stream` created via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  Ordinary array operations use the current
  [`mlx_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_device.md)
  instead.

## Value

An mlx object containing the pseudoinverse.

## Details

As of MLX 0.31.1, this operation only runs on CPU. Run it inside
[`with_device()`](https://hughjonesd.github.io/Rmlx/reference/with_device.md)
or
[`local_device()`](https://hughjonesd.github.io/Rmlx/reference/with_device.md),
or pass `device = "cpu"`.

## See also

[mlx.linalg.pinv](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.pinv)

## Examples

``` r
x <- mlx_matrix(c(1, 2, 3, 4), 2, 2)
pinv(x, device = "cpu")
#> mlx array [2 x 2]
#>   dtype: float32
#>   values:
#>      [,1]       [,2]
#> [1,]   -2  1.5000004
#> [2,]    1 -0.5000001
```
