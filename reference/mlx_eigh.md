# Eigen decomposition of Hermitian mlx arrays

Eigen decomposition of Hermitian mlx arrays

## Usage

``` r
mlx_eigh(x, uplo = c("L", "U"))
```

## Arguments

- x:

  An mlx matrix (2-dimensional array).

- uplo:

  Character string indicating which triangle to use ("L" or "U").

## Value

A list with components `values` and `vectors`.

## See also

[mlx.linalg.eigh](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.eigh)

## Examples

``` r
x <- mlx_matrix(c(2, 1, 1, 3), 2, 2)
mlx_eigh(x)
#> $values
#> mlx array [2]
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 1.381966 3.618034
#> 
#> $vectors
#> mlx array [2 x 2]
#>   dtype: float32
#>   device: gpu
#>   values:
#>            [,1]      [,2]
#> [1,] -0.8506508 0.5257311
#> [2,]  0.5257311 0.8506508
#> 
```
