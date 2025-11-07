# Leaky ReLU activation

Leaky ReLU activation

## Usage

``` r
mlx_leaky_relu(negative_slope = 0.01)
```

## Arguments

- negative_slope:

  Slope for negative values (default: 0.01).

## Value

An `mlx_module` applying Leaky ReLU activation.

## See also

[mlx.nn.LeakyReLU](https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.LeakyReLU)

## Examples

``` r
act <- mlx_leaky_relu(negative_slope = 0.1)
x <- as_mlx(matrix(c(-2, -1, 0, 1, 2), 5, 1))
mlx_forward(act, x)
#> mlx array [5 x 1]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1]
#> [1,] -0.2
#> [2,] -0.1
#> [3,]  0.0
#> [4,]  1.0
#> [5,]  2.0
```
