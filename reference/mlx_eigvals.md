# Eigenvalues of mlx arrays

Eigenvalues of mlx arrays

## Usage

``` r
mlx_eigvals(x)
```

## Arguments

- x:

  An mlx matrix (2-dimensional array).

## Value

An mlx array containing eigenvalues.

## See also

[mlx.linalg.eigvals](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.eigvals)

## Examples

``` r
x <- as_mlx(matrix(c(3, 1, 0, 2), 2, 2))
mlx_eigvals(x)
#> mlx array []
#>   dtype: complex64
#>   device: gpu
#>   values:
#> [1] 3+0i 2+0i
```
