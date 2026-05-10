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

  Execution target: supply `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  By default, many functions use the
  [`mlx_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_device.md)
  of their first argument.

## Value

A list with components `values` and `vectors`.

## Details

As of MLX 0.31.1, this operation only runs on CPU. Create or cast the
operands with `device = "cpu"` explicitly, or pass a `device = "cpu"`
argument. (Passing the argument won't affect the device of any mlx
object returned, just where this particular operation is run.)

## See also

[mlx.linalg.eigh](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.eigh)

## Examples

``` r
x <- mlx_matrix(c(2, 1, 1, 3), 2, 2, device = "cpu")
mlx_eigh(x)
#> $values
#> mlx array [2]
#>   dtype: float32
#>   device: cpu
#>   values:
#> [1] 1.381966 3.618034
#> 
#> $vectors
#> mlx array [2 x 2]
#>   dtype: float32
#>   device: cpu
#>   values:
#>            [,1]      [,2]
#> [1,] -0.8506508 0.5257311
#> [2,]  0.5257311 0.8506508
#> 
```
