# GELU activation

Gaussian Error Linear Unit activation function.

## Usage

``` r
mlx_gelu()
```

## Value

An `mlx_module` applying GELU activation.

## See also

[mlx.nn.GELU](https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.GELU)

## Examples

``` r
act <- mlx_gelu()
x <- as_mlx(matrix(c(-2, -1, 0, 1, 2), 5, 1))
mlx_forward(act, x)
#> mlx array [5 x 1]
#>   dtype: float32
#>   device: gpu
#>   values:
#>             [,1]
#> [1,] -0.04540229
#> [2,] -0.15880799
#> [3,]  0.00000000
#> [4,]  0.84119201
#> [5,]  1.95459771
```
