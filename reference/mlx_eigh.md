# Eigen decomposition of Hermitian mlx arrays

Eigen decomposition of Hermitian mlx arrays

## Usage

``` r
mlx_eigh(x, uplo = c("L", "U"), device = NULL)
```

## Arguments

- x:

  An mlx matrix (2-dimensional array).

- uplo:

  Character string indicating which triangle to use ("L" or "U").

- device:

  Execution target for APIs that expose a one-off device or stream
  override. Supply `"gpu"`, `"cpu"`, or an `mlx_stream` created via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  Ordinary array operations use the current
  [`mlx_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_device.md)
  instead.

## Value

A list with components `values` and `vectors`.

## Details

As of MLX 0.31.1, this operation only runs on CPU. Run it inside
[`with_device()`](https://hughjonesd.github.io/Rmlx/reference/with_device.md)
or
[`local_device()`](https://hughjonesd.github.io/Rmlx/reference/with_device.md),
or pass `device = "cpu"`.

## See also

[mlx.linalg.eigh](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.eigh)

## Examples

``` r
x <- mlx_matrix(c(2, 1, 1, 3), 2, 2)
mlx_eigh(x, device = "cpu")
#> $values
#> mlx array [2]
#>   dtype: float32
#>   values:
#> [1] 1.381966 3.618034
#> 
#> $vectors
#> mlx array [2 x 2]
#>   dtype: float32
#>   values:
#>            [,1]      [,2]
#> [1,] -0.8506508 0.5257311
#> [2,]  0.5257311 0.8506508
#> 
```
