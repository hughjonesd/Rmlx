# Compose modules sequentially

Compose modules sequentially

## Usage

``` r
mlx_sequential(...)
```

## Arguments

- ...:

  Modules to compose.

## Value

An `mlx_module`.

## See also

[mlx.nn.Sequential](https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.Sequential)

## Examples

``` r
set.seed(1)
net <- mlx_sequential(mlx_linear(2, 3), mlx_relu(), mlx_linear(3, 1))
input <- as_mlx(matrix(c(1, 2), 1, 2))
mlx_forward(net, input)
#> mlx array [1 x 1]
#>   dtype: float32
#>   device: gpu
#>   values:
#>          [,1]
#> [1,] 1.419647
```
