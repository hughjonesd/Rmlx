# LU factorization

Computes the LU factorization of a matrix.

## Usage

``` r
mlx_lu(x)
```

## Arguments

- x:

  An mlx array.

## Value

A list with components `p` (pivot indices), `l` (lower triangular), and
`u` (upper triangular). The relationship is `A = L[P, ] %*% U`.

## See also

[mlx.core.linalg.lu](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.core.linalg.lu)

## Examples

``` r
A <- as_mlx(matrix(rnorm(16), 4, 4))
lu_result <- mlx_lu(A)
P <- lu_result$p  # Pivot indices
L <- lu_result$l  # Lower triangular
U <- lu_result$u  # Upper triangular
```
