# Create arrays of ones on MLX devices

Create arrays of ones on MLX devices

## Usage

``` r
mlx_ones(
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

An mlx array filled with ones.

## See also

[mlx.core.ones](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.ones)

## Examples

``` r
ones <- with_device("cpu", mlx_ones(c(2, 2), dtype = "float64"))
ones_int <- mlx_ones(c(3, 3), dtype = "int32")
```
