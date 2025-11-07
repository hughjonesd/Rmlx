# Moore-Penrose pseudoinverse for MLX arrays

Moore-Penrose pseudoinverse for MLX arrays

## Usage

``` r
pinv(x)
```

## Arguments

- x:

  An mlx object or coercible matrix.

## Value

An mlx object containing the pseudoinverse.

## See also

[mlx.linalg.pinv](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.pinv)

## Examples

``` r
x <- as_mlx(matrix(c(1, 2, 3, 4), 2, 2))
pinv(x)
#> mlx array [2 x 2]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1]       [,2]
#> [1,]   -2  1.5000004
#> [2,]    1 -0.5000001
```
