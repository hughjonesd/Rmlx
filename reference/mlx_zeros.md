# Create arrays of zeros on MLX devices

Create arrays of zeros on MLX devices

## Usage

``` r
mlx_zeros(
  dim,
  dtype = c("float32", "float64", "int8", "int16", "int32", "int64", "uint8", "uint16",
    "uint32", "uint64", "bool", "complex64"),
  device = mlx_default_device()
)
```

## Arguments

- dim:

  Integer vector specifying array dimensions (shape).

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

An mlx array filled with zeros.

## See also

[mlx.core.zeros](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.zeros)

## Examples

``` r
zeros <- mlx_zeros(c(2, 3))
zeros_int <- mlx_zeros(c(2, 3), dtype = "int32")
```
