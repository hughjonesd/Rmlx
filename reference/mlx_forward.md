# Forward pass utility

Forward pass utility

## Usage

``` r
mlx_forward(module, x)
```

## Arguments

- module:

  An `mlx_module`.

- x:

  An mlx array.

## Value

Output array.

## See also

[mlx.nn.Module](https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.Module)

## Examples

``` r
set.seed(1)
layer <- mlx_linear(2, 1)
input <- as_mlx(matrix(c(1, 2), 1, 2))
mlx_forward(layer, input)
#> mlx array [1 x 1]
#>   dtype: float32
#>   device: gpu
#>   values:
#>            [,1]
#> [1,] -0.2591672
```
