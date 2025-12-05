# Compute matrix inverse

Computes the inverse of a square matrix. Note that as of MLX 0.30.0,
this runs on the CPU.

## Usage

``` r
mlx_inv(x)
```

## Arguments

- x:

  An mlx array.

## Value

The inverse of `x`.

## See also

[mlx.core.linalg.inv](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.core.linalg.inv)

## Examples

``` r
A <- mlx_matrix(c(4, 7, 2, 6), 2, 2)
A_inv <- mlx_inv(A)
# Verify: A %*% A_inv should be identity
A %*% A_inv
#> mlx array [2 x 2]
#>   dtype: float32
#>   device: cpu
#>   values:
#>              [,1] [,2]
#> [1,] 1.000000e+00    0
#> [2,] 4.768372e-07    1
```
