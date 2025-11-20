# Kronecker product for mlx arrays

Computes the Kronecker (tensor) product between two mlx arrays. Inputs
are automatically cast to a common dtype and device before evaluation.

## Usage

``` r
mlx_kron(a, b)
```

## Arguments

- a, b:

  Objects coercible to `mlx`.

## Value

An `mlx` array representing the Kronecker product.

## See also

[mlx.core.kron](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.kron)

## Examples

``` r
A <- mlx_matrix(1:4, 2, 2)
B <- mlx_matrix(c(0, 5, 6, 7), 2, 2)
mlx_kron(A, B)
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1] [,2] [,3] [,4]
#> [1,]    0    6    0   18
#> [2,]    5    7   15   21
#> [3,]    0   12    0   24
#> [4,]   10   14   20   28
```
