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

  MLX dtype to use. One of `"float32"`, `"float64"`, `"int8"`,
  `"int16"`, `"int32"`, `"int64"`, `"uint8"`, `"uint16"`, `"uint32"`,
  `"uint64"`, `"bool"`, or `"complex64"`.

- device:

  Execution target: supply `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  Defaults to the current
  [`mlx_default_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_default_device.md)
  unless noted otherwise (helpers that act on an existing array
  typically reuse that array's device or stream).

## Value

An mlx array filled with zeros.

## See also

[mlx.core.zeros](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.zeros)

## Examples

``` r
zeros <- mlx_zeros(c(2, 3))
zeros_int <- mlx_zeros(c(2, 3), dtype = "int32")
```
