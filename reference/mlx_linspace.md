# Evenly spaced ranges on MLX devices

`mlx_linspace()` creates `num` evenly spaced values from `start` to
`stop`, inclusive. Unlike
[`mlx_arange()`](https://hughjonesd.github.io/Rmlx/reference/mlx_arange.md),
you specify how many samples you want rather than the step size.

## Usage

``` r
mlx_linspace(
  start,
  stop,
  num = 50L,
  dtype = c("float32", "float64"),
  device = mlx_default_device()
)
```

## Arguments

- start:

  Starting value.

- stop:

  Final value (inclusive).

- num:

  Number of samples to generate.

- dtype:

  MLX dtype (`"float32"` or `"float64"`).

- device:

  Execution target: provide `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  Defaults to the current
  [`mlx_default_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_default_device.md).

## Value

A 1D mlx array.

## See also

[mlx.core.linspace](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.linspace)

## Examples

``` r
mlx_linspace(0, 1, num = 5)
#> mlx array [5]
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 0.00 0.25 0.50 0.75 1.00
```
