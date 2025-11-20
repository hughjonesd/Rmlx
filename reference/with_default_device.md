# Temporarily set the default MLX device

Temporarily set the default MLX device

## Usage

``` r
with_default_device(device, code)
```

## Arguments

- device:

  `"gpu"`, `"cpu"`, or an `mlx_stream` created via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).

- code:

  Expression to evaluate while `device` is active.

## Value

The result of evaluating `code`.

## See also

[mlx.core.default_device](https://ml-explore.github.io/mlx/build/html/python/metal.html)

## Examples

``` r
with_default_device("cpu", x <- mlx_vector(1:10))
```
