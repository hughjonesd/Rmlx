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

  Execution target for APIs that expose a one-off device or stream
  override. Supply `"gpu"`, `"cpu"`, or an `mlx_stream` created via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  Ordinary array operations use the current
  [`mlx_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_device.md)
  instead.

## Value

An mlx object containing the solution.

## Details

As of MLX 0.31.1, this operation only runs on CPU. Run it inside
[`with_device()`](https://hughjonesd.github.io/Rmlx/reference/with_device.md)
or
[`local_device()`](https://hughjonesd.github.io/Rmlx/reference/with_device.md),
or pass `device = "cpu"`.

## See also

[mlx.linalg.solve](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.solve)

## Examples

``` r
with_device("cpu", {
  a <- mlx_matrix(c(3, 1, 1, 2), 2, 2)
  b <- as_mlx(c(9, 8))
  solve(a, b)
})
#> mlx array [2]
#>   dtype: float32
#>   values:
#> [1] 2 3
```
