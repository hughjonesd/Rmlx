# Fill an mlx array with a constant value

Fill an mlx array with a constant value

## Usage

``` r
mlx_full(dim, value, dtype = NULL, device = mlx_default_device())
```

## Arguments

- dim:

  Integer vector specifying array dimensions (shape).

- value:

  Scalar value used to fill the array. Numeric, logical, or complex.

- dtype:

  Data type string. Supported types include:

  - Floating point: `"float32"`, `"float64"`

  - Integer: `"int8"`, `"int16"`, `"int32"`, `"int64"`, `"uint8"`,
    `"uint16"`, `"uint32"`, `"uint64"`

  - Other: `"bool"`, `"complex64"`

  Not all functions support all types. See individual function
  documentation.

- device:

  Execution target: supply `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  Defaults to the current
  [`mlx_default_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_default_device.md)
  unless noted otherwise (helpers that act on an existing array
  typically reuse that array's device or stream).

## Value

An mlx array filled with the supplied value.

## See also

[mlx.core.full](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.full)

## Examples

``` r
filled <- mlx_full(c(2, 2), 3.14)
complex_full <- mlx_full(c(2, 2), 1+2i, dtype = "complex64")
```
