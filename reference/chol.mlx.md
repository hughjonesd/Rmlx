# Cholesky decomposition for mlx arrays

If `x` is not symmetric positive semi-definite, "behaviour is undefined"
according to the MLX documentation.

## Usage

``` r
# S3 method for class 'mlx'
chol(x, pivot = FALSE, ..., device = NULL)
```

## Arguments

- x:

  An mlx matrix (2-dimensional array).

- pivot:

  Ignored; pivoted decomposition is not supported.

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

Upper-triangular Cholesky factor as an mlx matrix.

## Details

As of MLX 0.31.1, this operation only runs on CPU. Run it inside
[`with_device()`](https://hughjonesd.github.io/Rmlx/reference/with_device.md)
or
[`local_device()`](https://hughjonesd.github.io/Rmlx/reference/with_device.md),
or pass `device = "cpu"`.

## See also

[mlx.linalg.cholesky](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.cholesky)

## Examples

``` r
x <- mlx_matrix(c(4, 1, 1, 3), 2, 2)
chol(x, device = "cpu")
#> mlx array [2 x 2]
#>   dtype: float32
#>   values:
#>      [,1]     [,2]
#> [1,]    2 0.500000
#> [2,]    0 1.658312
```
