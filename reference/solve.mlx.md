# Solve a system of linear equations

Solve a system of linear equations

## Usage

``` r
# S3 method for class 'mlx'
solve(a, b = NULL, ..., device = NULL)
```

## Arguments

- a:

  An mlx matrix of coefficients.

- b:

  An mlx vector or matrix (the right-hand side). If omitted, computes
  the matrix inverse.

- ...:

  Additional arguments forwarded to the corresponding base R
  implementation for signature compatibility.

- device:

  Execution target: supply `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  By default, many functions use the
  [`mlx_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_device.md)
  of their first argument.

## Value

An mlx object containing the solution.

## Details

As of MLX 0.31.1, this operation only runs on CPU. Create or cast the
operands with `device = "cpu"` explicitly, or pass a `device = "cpu"`
argument. (Passing the argument won't affect the device of any mlx
object returned, just where this particular operation is run.)

## See also

[mlx.linalg.solve](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.solve)

## Examples

``` r
a <- mlx_matrix(c(3, 1, 1, 2), 2, 2, device = "cpu")
b <- as_mlx(c(9, 8), device = "cpu")
solve(a, b)
#> mlx array [2]
#>   dtype: float32
#>   device: cpu
#>   values:
#> [1] 2 3
```
