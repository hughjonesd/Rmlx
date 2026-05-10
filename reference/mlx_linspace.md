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

  Data type string. Supported types include:

  - Floating point: `"float32"`, `"float64"`

  - Integer: `"int8"`, `"int16"`, `"int32"`, `"int64"`, `"uint8"`,
    `"uint16"`, `"uint32"`, `"uint64"`

  - Other: `"bool"`, `"complex64"`

  `float64` arrays are CPU-only. Use `device = "cpu"` when creating or
  casting to `float64`, and cast back to `float32` before using the GPU.
  Not all functions support all types. See individual function
  documentation.

- device:

  Execution target: supply `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  By default, many functions use the
  [`mlx_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_device.md)
  of their first argument.

## Value

A 1D mlx array.

## See also

[mlx.core.linspace](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.linspace)

## Examples

``` r
mlx_linspace(0, 1, num = 5)
#> mlx array [5]
#>   dtype: float32
#>   device: cpu
#>   values:
#> [1] 0.00 0.25 0.50 0.75 1.00
```
