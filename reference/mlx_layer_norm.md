# Layer normalization

Normalizes inputs across the feature dimension.

## Usage

``` r
mlx_layer_norm(normalized_shape, eps = 1e-05, device = mlx_default_device())
```

## Arguments

- normalized_shape:

  Size of the feature dimension to normalize.

- eps:

  Small constant for numerical stability (default: 1e-5).

- device:

  Execution target: supply `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  Default:
  [`mlx_default_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_default_device.md).

## Value

An `mlx_module` applying layer normalization.

## See also

[mlx.nn.LayerNorm](https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.LayerNorm)

## Examples

``` r
set.seed(1)
ln <- mlx_layer_norm(4)
x <- as_mlx(matrix(rnorm(12), 3, 4))
mlx_forward(ln, x)
#> mlx array [3 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>            [,1]       [,2]       [,3]       [,4]
#> [1,] -1.0668312  1.5259182 0.23306273 -0.6921500
#> [2,] -0.9833397 -0.7005272 0.09211669  1.5917501
#> [3,] -1.0064702 -0.9834564 1.13609314  0.8538334
```
