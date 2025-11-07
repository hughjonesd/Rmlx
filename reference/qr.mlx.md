# QR decomposition for mlx arrays

QR decomposition for mlx arrays

## Usage

``` r
# S3 method for class 'mlx'
qr(x, tol = 1e-07, LAPACK = FALSE, ...)
```

## Arguments

- x:

  An mlx matrix (2-dimensional array).

- tol:

  Ignored; custom tolerances are not supported.

- LAPACK:

  Ignored; set to `FALSE`.

- ...:

  Additional arguments (unused).

## Value

A list with components `Q` and `R`, each an mlx matrix.

## See also

[mlx.linalg.qr](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.qr)

## Examples

``` r
x <- as_mlx(matrix(c(1, 2, 3, 4, 5, 6), 3, 2))
qr(x)
#> $Q
#> mlx array [3 x 2]
#>   dtype: float32
#>   device: gpu
#>   values:
#>            [,1]       [,2]
#> [1,] -0.2672611  0.8728715
#> [2,] -0.5345225  0.2182179
#> [3,] -0.8017837 -0.4364358
#> 
#> $R
#> mlx array [2 x 2]
#>   dtype: float32
#>   device: gpu
#>   values:
#>           [,1]      [,2]
#> [1,] -3.741657 -8.552359
#> [2,]  0.000000  1.963961
#> 
#> attr(,"class")
#> [1] "mlx_qr" "list"  
```
