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

  Execution target: supply `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  By default, many functions use the
  [`mlx_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_device.md)
  of their first argument.

## Value

Upper-triangular Cholesky factor as an mlx matrix.

## Details

As of MLX 0.31.1, this operation only runs on CPU. Create or cast the
operands with `device = "cpu"` explicitly, or pass a `device = "cpu"`
argument. (Passing the argument won't affect the device of any mlx
object returned, just where this particular operation is run.)

## See also

[mlx.linalg.cholesky](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.cholesky)

## Examples

``` r
x <- mlx_matrix(c(4, 1, 1, 3), 2, 2, device = "cpu")
chol(x)
#> mlx array [2 x 2]
#>   dtype: float32
#>   device: cpu
#>   values:
#>      [,1]     [,2]
#> [1,]    2 0.500000
#> [2,]    0 1.658312
```
