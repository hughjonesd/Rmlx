# Solve triangular systems with mlx arrays

Solve triangular systems with mlx arrays

## Usage

``` r
mlx_solve_triangular(a, b, upper = FALSE, device = NULL)

backsolve(r, x, k = NULL, upper.tri = TRUE, transpose = FALSE, ...)

# Default S3 method
backsolve(r, x, k = NULL, upper.tri = TRUE, transpose = FALSE, ...)

# S3 method for class 'mlx'
backsolve(r, x, k = NULL, upper.tri = TRUE, transpose = FALSE, ...)
```

## Arguments

- a:

  An mlx triangular matrix.

- b:

  Right-hand side matrix or vector.

- upper:

  Logical; if `TRUE`, `a` is upper triangular, otherwise lower.

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

As of MLX 0.31.1, this operation only runs on CPU. Create or cast the
operands with `device = "cpu"` explicitly, or pass a `device = "cpu"`
argument. (Passing the argument won't affect the device of any mlx
object returned, just where this particular operation is run.)

## See also

[mlx.linalg.solve_triangular](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.solve_triangular)

## Examples

``` r
a <- mlx_matrix(c(2, 1, 0, 3), 2, 2, device = "cpu")
b <- mlx_matrix(c(1, 5), 2, 1, device = "cpu")
mlx_solve_triangular(a, b, upper = FALSE)
#> mlx array [2 x 1]
#>   dtype: float32
#>   device: cpu
#>   values:
#>      [,1]
#> [1,]  0.5
#> [2,]  1.5
```
