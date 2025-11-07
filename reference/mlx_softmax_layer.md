# Softmax activation

Softmax activation

## Usage

``` r
mlx_softmax_layer(axis = -1L)
```

## Arguments

- axis:

  Axis along which to apply softmax (default: -1, last axis).

## Value

An `mlx_module` applying softmax activation.

## See also

[mlx.nn.Softmax](https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.Softmax)

## Examples

``` r
act <- mlx_softmax_layer()
x <- as_mlx(matrix(1:6, 2, 3))
mlx_forward(act, x)
#> mlx array [2 x 3]
#>   dtype: float32
#>   device: gpu
#>   values:
#>            [,1]      [,2]      [,3]
#> [1,] 0.01587624 0.1173104 0.8668134
#> [2,] 0.01587624 0.1173104 0.8668134
```
