# Solve triangular systems with mlx arrays

Solve triangular systems with mlx arrays

## Usage

``` r
mlx_solve_triangular(a, b, upper = FALSE, device = NULL)

backsolve(r, x, k = NULL, upper.tri = TRUE, transpose = FALSE, ...)

# Default S3 method
backsolve(r, x, k = NULL, upper.tri = TRUE, transpose = FALSE, ...)

# S3 method for class 'mlx'
backsolve(
  r,
  x,
  k = NULL,
  upper.tri = TRUE,
  transpose = FALSE,
  ...,
  device = NULL
)
```

## Arguments

- a:

  An mlx triangular matrix.

- b:

  Right-hand side matrix or vector.

- upper:

  Logical; if `TRUE`, `a` is upper triangular, otherwise lower.

- device:

  Execution target for APIs that expose a one-off device or stream
  override. Supply `"gpu"`, `"cpu"`, or an `mlx_stream` created via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  Ordinary array operations use the current
  [`mlx_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_device.md)
  instead.

- r:

  Triangular system matrix passed to `backsolve()`.

- x:

  Right-hand side supplied to `backsolve()`.

- k:

  Number of columns of `r` to use.

- upper.tri:

  Logical; indicates if `r` is upper triangular.

- transpose:

  Logical; if `TRUE`, solve `t(r) %*% x = b`.

- ...:

  Additional arguments forwarded to the corresponding base R
  implementation for signature compatibility.

## Value

An mlx array solution.

## Details

As of MLX 0.31.1, this operation only runs on CPU. Run it inside
[`with_device()`](https://hughjonesd.github.io/Rmlx/reference/with_device.md)
or
[`local_device()`](https://hughjonesd.github.io/Rmlx/reference/with_device.md),
or pass `device = "cpu"`.

## See also

[mlx.linalg.solve_triangular](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.solve_triangular)

## Examples

``` r
a <- mlx_matrix(c(2, 1, 0, 3), 2, 2)
b <- mlx_matrix(c(1, 5), 2, 1)
mlx_solve_triangular(a, b, upper = FALSE, device = "cpu")
#> mlx array [2 x 1]
#>   dtype: float32
#>   values:
#>      [,1]
#> [1,]  0.5
#> [2,]  1.5
```
