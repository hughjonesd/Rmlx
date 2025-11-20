# Singular value decomposition for mlx arrays

Note that mlx's svd returns "full" SVD, with U and V' both square
matrices. This is different from R's implementation.

## Usage

``` r
# S3 method for class 'mlx'
svd(x, nu = min(n, p), nv = min(n, p), ...)
```

## Arguments

- x:

  An mlx matrix (2-dimensional array).

- nu:

  Number of left singular vectors to return (0 or `min(dim(x))`).

- nv:

  Number of right singular vectors to return (0 or `min(dim(x))`).

- ...:

  Additional arguments (unused).

## Value

A list with components `d`, `u`, and `v`.

## See also

[mlx.linalg.svd](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.svd)

## Examples

``` r
x <- mlx_matrix(c(1, 0, 0, 2), 2, 2)
svd(x)
#> $d
#> mlx array [2]
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 2 1
#> 
#> $u
#> mlx array [2 x 2]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1] [,2]
#> [1,]    0    1
#> [2,]    1    0
#> 
#> $v
#> mlx array [2 x 2]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1] [,2]
#> [1,]    0    1
#> [2,]    1    0
#> 
```
