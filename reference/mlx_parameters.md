# Collect parameters from modules

Collect parameters from modules

## Usage

``` r
mlx_parameters(module)
```

## Arguments

- module:

  An `mlx_module` or list of modules.

## Value

A list of `mlx_param` objects.

## See also

[mlx.nn.Module.parameters](https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.Module.parameters)

## Examples

``` r
set.seed(1)
layer <- mlx_linear(2, 1)
mlx_parameters(layer)
#> [[1]]
#> $env
#> <environment: 0x11ae629c8>
#> 
#> $name
#> [1] "weight"
#> 
#> attr(,"class")
#> [1] "mlx_param"
#> 
#> [[2]]
#> $env
#> <environment: 0x11ae629c8>
#> 
#> $name
#> [1] "bias"
#> 
#> attr(,"class")
#> [1] "mlx_param"
#> 
```
