# Identity matrices on MLX devices

Identity matrices on MLX devices

## Usage

``` r
mlx_identity(n, dtype = c("float32", "float64"))
```

## Arguments

- n:

  Size of the square matrix.

- dtype:

  Data type string. Supported types include:

  - Floating point: `"float32"`, `"float64"`

  - Integer: `"int8"`, `"int16"`, `"int32"`, `"int64"`, `"uint8"`,
    `"uint16"`, `"uint32"`, `"uint64"`

  - Other: `"bool"`, `"complex64"`

  Not all functions support all types. See individual function
  documentation.

## Value

An mlx identity matrix.

## Details

MLX does not support `float64` operations on GPU. When this function
creates a `float64` array or converts one back to R, Rmlx temporarily
switches only that internal creation or layout work to CPU. Later
operations on the returned array still use the current
[`mlx_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_device.md).

## See also

[mlx.core.identity](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.identity)

## Examples

``` r
I4 <- mlx_identity(4)
```
