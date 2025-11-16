# Temporarily set the default MLX device

Temporarily set the default MLX device

## Usage

``` r
with_default_device(device, code)
```

## Arguments

- device:

  Execution target: supply `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  Defaults to the current
  [`mlx_default_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_default_device.md)
  unless noted otherwise (helpers that act on an existing array
  typically reuse that array's device or stream).

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
