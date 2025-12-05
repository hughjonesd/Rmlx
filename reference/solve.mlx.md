# Solve a system of linear equations

Note that as of MLX 0.30.0,
[`solve()`](https://rdrr.io/r/base/solve.html) runs on the CPU.

## Usage

``` r
# S3 method for class 'mlx'
solve(a, b = NULL, ...)
```

## Arguments

- a:

  An mlx matrix (the coefficient matrix)

- b:

  An mlx vector or matrix (the right-hand side). If omitted, computes
  the matrix inverse.

- ...:

  Additional arguments forwarded to the corresponding base R
  implementation for signature compatibility.

## Value

An mlx object containing the solution.

## See also

[mlx.linalg.solve](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.solve)

## Examples

``` r
a <- mlx_matrix(c(3, 1, 1, 2), 2, 2)
b <- as_mlx(c(9, 8))
solve(a, b)
#> mlx array [2]
#>   dtype: float32
#>   device: cpu
#>   values:
#> [1] 2 3
```
