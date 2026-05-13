# Zeros shaped like an existing mlx array

`mlx_zeros_like()` mirrors
[`mlx.core.zeros_like()`](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.zeros_like):
it creates a zero-filled array matching the source array's shape.
Optionally override the dtype or dtype.

## Usage

``` r
mlx_zeros_like(x, dtype = NULL)
```

## Arguments

- x:

  An mlx array.

- dtype:

  Data type string. Supported types include:

  - Floating point: `"float32"`, `"float64"`

  - Integer: `"int8"`, `"int16"`, `"int32"`, `"int64"`, `"uint8"`,
    `"uint16"`, `"uint32"`, `"uint64"`

  - Other: `"bool"`, `"complex64"`

  Not all functions support all types. See individual function
  documentation.

## Value

An mlx array of zeros matching `x`.

## Details

MLX does not support `float64` operations on GPU. When this function
creates a `float64` array or converts one back to R, Rmlx temporarily
switches only that internal creation or layout work to CPU. Later
operations on the returned array still use the current
[`mlx_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_device.md).

## See also

[mlx.core.zeros_like](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.zeros_like)

## Examples

``` r
base <- mlx_ones(c(2, 2))
mlx_zeros_like(base)
#> mlx array [2 x 2]
#>   dtype: float32
#>   values:
#>      [,1] [,2]
#> [1,]    0    0
#> [2,]    0    0
```
