# Create arrays of zeros on MLX devices

Create arrays of zeros on MLX devices

## Usage

``` r
mlx_zeros(
  dim,
  dtype = c("float32", "float64", "int8", "int16", "int32", "int64", "uint8", "uint16",
    "uint32", "uint64", "bool", "complex64")
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

  Not all functions support all types. See individual function
  documentation.

## Value

An mlx array filled with zeros.

## Details

MLX does not support `float64` operations on GPU. When this function
creates a `float64` array or converts one back to R, Rmlx temporarily
switches only that internal creation or layout work to CPU. Later
operations on the returned array still use the current
[`mlx_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_device.md).

## See also

[mlx.core.zeros](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.zeros)

## Examples

``` r
zeros <- mlx_zeros(c(2, 3))
zeros_int <- mlx_zeros(c(2, 3), dtype = "int32")
```
