# Compute triangular matrix inverse

Computes the inverse of a triangular matrix.

## Usage

``` r
mlx_tri_inv(x, upper = FALSE)
```

## Arguments

- x:

  An mlx array.

- upper:

  Logical; if `TRUE`, `x` is upper triangular, otherwise lower
  triangular.

## Value

The inverse of the triangular matrix `x`.

## See also

[mlx.core.linalg.tri_inv](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.core.linalg.tri_inv)

## Examples

``` r
# Lower triangular matrix
L <- as_mlx(matrix(c(1, 2, 0, 3, 0, 0, 4, 5, 6), 3, 3, byrow = TRUE))
L_inv <- mlx_tri_inv(L, upper = FALSE)
```
