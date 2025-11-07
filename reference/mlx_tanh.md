# Tanh activation

Tanh activation

## Usage

``` r
mlx_tanh()
```

## Value

An `mlx_module` applying hyperbolic tangent activation.

## See also

[mlx.nn.Tanh](https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.Tanh)

## Examples

``` r
act <- mlx_tanh()
x <- as_mlx(matrix(c(-2, -1, 0, 1, 2), 5, 1))
mlx_forward(act, x)
#> mlx array [5 x 1]
#>   dtype: float32
#>   device: gpu
#>   values:
#>            [,1]
#> [1,] -0.9640276
#> [2,] -0.7615941
#> [3,]  0.0000000
#> [4,]  0.7615941
#> [5,]  0.9640276
```
