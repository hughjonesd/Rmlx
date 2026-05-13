# Singular value decomposition for mlx arrays

Note that mlx's svd returns "full" SVD, with U and V' both square
matrices. This is different from R's implementation.

## Usage

``` r
# S3 method for class 'mlx'
svd(x, nu = min(n, p), nv = min(n, p), ..., device = NULL)
```

## Arguments

- x:

  An mlx matrix (2-dimensional array).

- nu:

  Number of left singular vectors to return (0 or `min(dim(x))`).

- nv:

  Number of right singular vectors to return (0 or `min(dim(x))`).

- ...:

  Additional arguments; ignored.

- device:

  Execution target for APIs that expose a one-off device or stream
  override. Supply `"gpu"`, `"cpu"`, or an `mlx_stream` created via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  Ordinary array operations use the current
  [`mlx_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_device.md)
  instead.

## Value

A list with components `d`, `u`, and `v`.

## Details

As of MLX 0.31.1, this operation only runs on CPU. Run it inside
[`with_device()`](https://hughjonesd.github.io/Rmlx/reference/with_device.md)
or
[`local_device()`](https://hughjonesd.github.io/Rmlx/reference/with_device.md),
or pass `device = "cpu"`.

## See also

[mlx.linalg.svd](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.svd)

## Examples

``` r
x <- mlx_matrix(c(1, 0, 0, 2), 2, 2)
svd(x, device = "cpu")
#> $d
#> mlx array [2]
#>   dtype: float32
#>   values:
#> [1] 2 1
#> 
#> $u
#> mlx array [2 x 2]
#>   dtype: float32
#>   values:
#>      [,1] [,2]
#> [1,]    0    1
#> [2,]    1    0
#> 
#> $v
#> mlx array [2 x 2]
#>   dtype: float32
#>   values:
#>      [,1] [,2]
#> [1,]    0    1
#> [2,]    1    0
#> 
```
