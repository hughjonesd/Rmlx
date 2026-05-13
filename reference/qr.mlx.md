# QR decomposition for mlx arrays

QR decomposition for mlx arrays

## Usage

``` r
# S3 method for class 'mlx'
qr(x, tol = 1e-07, LAPACK = FALSE, ..., device = NULL)
```

## Arguments

- x:

  An mlx matrix (2-dimensional array).

- tol:

  Ignored; custom tolerances are not supported.

- LAPACK:

  Ignored; set to `FALSE`.

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

A list with components `Q` and `R`, each an mlx matrix.

## Details

As of MLX 0.31.1, this operation only runs on CPU. Run it inside
[`with_device()`](https://hughjonesd.github.io/Rmlx/reference/with_device.md)
or
[`local_device()`](https://hughjonesd.github.io/Rmlx/reference/with_device.md),
or pass `device = "cpu"`.

## See also

[mlx.linalg.qr](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.qr)

## Examples

``` r
x <- mlx_matrix(c(1, 2, 3, 4, 5, 6), 3, 2)
qr(x, device = "cpu")
#> $Q
#> mlx array [3 x 2]
#>   dtype: float32
#>   values:
#>            [,1]       [,2]
#> [1,] -0.2672611  0.8728715
#> [2,] -0.5345225  0.2182179
#> [3,] -0.8017837 -0.4364358
#> 
#> $R
#> mlx array [2 x 2]
#>   dtype: float32
#>   values:
#>           [,1]      [,2]
#> [1,] -3.741657 -8.552359
#> [2,]  0.000000  1.963961
#> 
#> attr(,"class")
#> [1] "mlx_qr" "list"  
```
