# Temporarily set the default MLX device

Temporarily set the default MLX device

## Usage

``` r
with_default_device(device = c("gpu", "cpu"), code)
```

## Arguments

- device:

  Device to use (`"gpu"` or `"cpu"`).

- code:

  Expression to evaluate while `device` is active.

## Value

The result of evaluating `code`.

## See also

[mlx.core.default_device](https://ml-explore.github.io/mlx/build/html/python/metal.html)

## Examples

``` r
old <- mlx_default_device()
with_default_device("cpu", mlx_default_device())
#> [1] "cpu"
mlx_default_device(old)
#> [1] "gpu"
```
