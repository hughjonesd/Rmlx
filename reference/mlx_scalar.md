# Construct MLX scalars

Construct MLX scalars

## Usage

``` r
mlx_scalar(value, dtype = NULL, device = mlx_default_device())
```

## Arguments

- value:

  Single value (numeric, logical, or complex).

- dtype:

  Optional MLX dtype. Defaults to `"float32"` for numeric input,
  `"bool"` for logical, and `"complex64"` for complex.

- device:

  Execution target: supply `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  Defaults to the current
  [`mlx_default_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_default_device.md)
  unless noted otherwise (helpers that act on an existing array
  typically reuse that array's device or stream).

## Value

A dimensionless `mlx` scalar.
