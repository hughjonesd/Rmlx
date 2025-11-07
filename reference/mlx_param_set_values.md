# Assign arrays back to parameters

Assign arrays back to parameters

## Usage

``` r
mlx_param_set_values(params, values)
```

## Arguments

- params:

  A list of `mlx_param`.

- values:

  A list of arrays.

## See also

[mlx.nn.Module.update](https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.Module.update)

## Examples

``` r
set.seed(1)
layer <- mlx_linear(2, 1)
params <- mlx_parameters(layer)
current <- mlx_param_values(params)
mlx_param_set_values(params, current)
```
