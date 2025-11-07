# Retrieve parameter arrays

Retrieve parameter arrays

## Usage

``` r
mlx_param_values(params)
```

## Arguments

- params:

  A list of `mlx_param`.

## Value

List of mlx arrays.

## See also

[mlx.nn.Module.parameters](https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.Module.parameters)

## Examples

``` r
set.seed(1)
layer <- mlx_linear(2, 1)
vals <- mlx_param_values(mlx_parameters(layer))
```
