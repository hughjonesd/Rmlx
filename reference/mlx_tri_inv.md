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

## Details

**Note:** MLX may crash if `x` is not triangular.

## See also

[mlx.core.linalg.tri_inv](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.core.linalg.tri_inv)

## Examples

``` r
# Lower triangular matrix
L <- mlx_matrix(c(1:3, 0, 4:5, 0, 0, 6), 3, 3)
mlx_tri_inv(L, upper = FALSE)
#> mlx array [3 x 3]
#>   dtype: float32
#>   device: gpu
#>   values:
#>             [,1]       [,2]      [,3]
#> [1,]  1.00000000  0.0000000 0.0000000
#> [2,] -0.50000000  0.2500000 0.0000000
#> [3,] -0.08333334 -0.2083333 0.1666667
```
