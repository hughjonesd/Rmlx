# Compute triangular matrix inverse

Computes the inverse of a triangular matrix.

## Usage

``` r
mlx_tri_inv(x, upper = FALSE, device = NULL)
```

## Arguments

- x:

  An mlx array.

- upper:

  Logical; if `TRUE`, `x` is upper triangular, otherwise lower
  triangular.

- device:

  Execution target for APIs that expose a one-off device or stream
  override. Supply `"gpu"`, `"cpu"`, or an `mlx_stream` created via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  Ordinary array operations use the current
  [`mlx_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_device.md)
  instead.

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
mlx_tri_inv(L, upper = FALSE, device = "cpu")
#> mlx array [3 x 3]
#>   dtype: float32
#>   values:
#>             [,1]       [,2]      [,3]
#> [1,]  1.00000000  0.0000000 0.0000000
#> [2,] -0.50000000  0.2500000 0.0000000
#> [3,] -0.08333334 -0.2083333 0.1666667
```
