# Rectified linear activation module

Rectified linear activation module

## Usage

``` r
mlx_relu()
```

## Value

An `mlx_module` applying ReLU.

## See also

[mlx.nn.ReLU](https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.ReLU)

## Examples

``` r
act <- mlx_relu()
x <- as_mlx(matrix(c(-1, 0, 2), 3, 1))
mlx_forward(act, x)
#> mlx array [3 x 1]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1]
#> [1,]    0
#> [2,]    0
#> [3,]    2
```
