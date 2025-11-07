# Sigmoid activation

Sigmoid activation

## Usage

``` r
mlx_sigmoid()
```

## Value

An `mlx_module` applying sigmoid activation.

## See also

[mlx.nn.Sigmoid](https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.Sigmoid)

## Examples

``` r
act <- mlx_sigmoid()
x <- as_mlx(matrix(c(-2, -1, 0, 1, 2), 5, 1))
mlx_forward(act, x)
#> mlx array [5 x 1]
#>   dtype: float32
#>   device: gpu
#>   values:
#>           [,1]
#> [1,] 0.1192029
#> [2,] 0.2689414
#> [3,] 0.5000000
#> [4,] 0.7310586
#> [5,] 0.8807970
```
