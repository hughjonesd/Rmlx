# Eigenvalues of Hermitian mlx arrays

Eigenvalues of Hermitian mlx arrays

## Usage

``` r
mlx_eigvalsh(x, uplo = c("L", "U"))
```

## Arguments

- x:

  An mlx matrix (2-dimensional array).

- uplo:

  Character string indicating which triangle to use ("L" or "U").

## Value

An mlx array containing eigenvalues.

## See also

[mlx.linalg.eigvalsh](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.eigvalsh)

## Examples

``` r
x <- mlx_matrix(c(2, 1, 1, 3), 2, 2)
mlx_eigvalsh(x)
#> mlx array [2]
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 1.381966 3.618034
```
