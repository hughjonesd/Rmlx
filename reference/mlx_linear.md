# Create a learnable linear transformation

Create a learnable linear transformation

## Usage

``` r
mlx_linear(
  in_features,
  out_features,
  bias = TRUE,
  device = mlx_default_device()
)
```

## Arguments

- in_features:

  Number of input features.

- out_features:

  Number of output features.

- bias:

  Should a bias term be included?

- device:

  Execution target: supply `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  Default:
  [`mlx_default_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_default_device.md).

## Value

An object of class `mlx_module`.

## See also

[mlx.nn.Linear](https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.Linear)

## Examples

``` r
set.seed(1)
layer <- mlx_linear(3, 2)
x <- as_mlx(matrix(1:6, 2, 3))
mlx_forward(layer, x)
#> mlx array [2 x 2]
#>   dtype: float32
#>   device: gpu
#>   values:
#>           [,1]       [,2]
#> [1,] -3.473105 -1.2398810
#> [2,] -4.516946 -0.3382074
```
