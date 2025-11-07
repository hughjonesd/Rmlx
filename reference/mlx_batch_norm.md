# Batch normalization

Normalizes inputs across the batch dimension.

## Usage

``` r
mlx_batch_norm(
  num_features,
  eps = 1e-05,
  momentum = 0.1,
  device = mlx_default_device()
)
```

## Arguments

- num_features:

  Number of feature channels.

- eps:

  Small constant for numerical stability (default: 1e-5).

- momentum:

  Momentum for running statistics (default: 0.1).

- device:

  Execution target: supply `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  Default:
  [`mlx_default_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_default_device.md).

## Value

An `mlx_module` applying batch normalization.

## See also

[mlx.nn.BatchNorm](https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.BatchNorm)

## Examples

``` r
set.seed(1)
bn <- mlx_batch_norm(4)
x <- as_mlx(matrix(rnorm(12), 3, 4))
mlx_forward(bn, x)
#> mlx array [3 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>            [,1]        [,2]       [,3]       [,4]
#> [1,] -0.4556868  1.24383128 -1.0877743 -1.1186367
#> [2,]  1.3872330 -0.03912285  1.3256620  1.3086261
#> [3,] -0.9315463 -1.20470834 -0.2378886 -0.1899893
```
