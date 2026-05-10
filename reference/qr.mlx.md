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

  Execution target: supply `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  By default, many functions use the
  [`mlx_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_device.md)
  of their first argument.

## Value

A list with components `Q` and `R`, each an mlx matrix.

## Details

As of MLX 0.31.1, this operation only runs on CPU. Create or cast the
operands with `device = "cpu"` explicitly, or pass a `device = "cpu"`
argument. (Passing the argument won't affect the device of any mlx
object returned, just where this particular operation is run.)

## See also

[mlx.linalg.qr](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.qr)

## Examples

``` r
x <- mlx_matrix(c(1, 2, 3, 4, 5, 6), 3, 2, device = "cpu")
qr(x)
#> $Q
#> mlx array [3 x 2]
#>   dtype: float32
#>   device: cpu
#>   values:
#>            [,1]       [,2]
#> [1,] -0.2672611  0.8728715
#> [2,] -0.5345225  0.2182179
#> [3,] -0.8017837 -0.4364358
#> 
#> $R
#> mlx array [2 x 2]
#>   dtype: float32
#>   device: cpu
#>   values:
#>           [,1]      [,2]
#> [1,] -3.741657 -8.552359
#> [2,]  0.000000  1.963961
#> 
#> attr(,"class")
#> [1] "mlx_qr" "list"  
```
