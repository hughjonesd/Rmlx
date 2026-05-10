# Eigenvalues of mlx arrays

Eigenvalues of mlx arrays

## Usage

``` r
mlx_eigvals(x, device = NULL)
```

## Arguments

- x:

  An mlx matrix (2-dimensional array).

- device:

  Execution target: supply `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  By default, many functions use the
  [`mlx_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_device.md)
  of their first argument.

## Value

An mlx array containing eigenvalues.

## Details

As of MLX 0.31.1, this operation only runs on CPU. Create or cast the
operands with `device = "cpu"` explicitly, or pass a `device = "cpu"`
argument. (Passing the argument won't affect the device of any mlx
object returned, just where this particular operation is run.)

## See also

[mlx.linalg.eigvals](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.eigvals)

## Examples

``` r
x <- mlx_matrix(c(3, 1, 0, 2), 2, 2, device = "cpu")
mlx_eigvals(x)
#> mlx array [2]
#>   dtype: complex64
#>   device: cpu
#>   values:
#> [1] 3+0i 2+0i
```
