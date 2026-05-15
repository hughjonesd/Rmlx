# Construct an MLX array from R data

`mlx_array()` is a low-level constructor that skips
[`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md)'s
type inference and dimension guessing. Supply the raw payload vector
plus an explicit shape and it pipes the data straight into MLX.

## Usage

``` r
mlx_array(data, dim, dtype = NULL, dimnames = NULL)
```

## Arguments

- data:

  Numeric, logical, or complex vector. `data` is recycled to match
  dimensions according to R rules (but with an error if it doesn't tile
  into the dimensions exactly).

- dim:

  Integer vector of array dimensions. Set `dim = integer(0)` for a
  scalar, in which case `data` must be length 1.

- dtype:

  Data type string. Supported types include:

  - Floating point: `"float32"`, `"float64"`

  - Integer: `"int8"`, `"int16"`, `"int32"`, `"int64"`, `"uint8"`,
    `"uint16"`, `"uint32"`, `"uint64"`

  - Other: `"bool"`, `"complex64"`

  Not all functions support all types. See individual function
  documentation.

- dimnames:

  Optional list of character vectors naming each dimension.

## Value

An `mlx` array with the requested shape.

## Details

MLX does not support `float64` operations on GPU. When this function
creates a `float64` array or converts one back to R, Rmlx temporarily
switches only that internal creation or layout work to CPU. Later
operations on the returned array still use the current
[`mlx_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_device.md).

## Examples

``` r
payload <- runif(6)
mlx_array(payload, dim = c(2, 3))
#> mlx array [2 x 3]
#>   dtype: float32
#>   values:
#>           [,1]      [,2]      [,3]
#> [1,] 0.7064338 0.1803388 0.6801629
#> [2,] 0.9485766 0.2168999 0.4988456
```
