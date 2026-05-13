# Identity-like matrices on MLX devices

Identity-like matrices on MLX devices

## Usage

``` r
mlx_eye(n, m = n, k = 0L, dtype = c("float32", "float64"))
```

## Arguments

- n:

  Number of rows.

- m:

  Optional number of columns (defaults to `n`).

- k:

  Diagonal index: `0` is the main diagonal, positive values shift
  upward, negative values shift downward.

- dtype:

  Data type string. Supported types include:

  - Floating point: `"float32"`, `"float64"`

  - Integer: `"int8"`, `"int16"`, `"int32"`, `"int64"`, `"uint8"`,
    `"uint16"`, `"uint32"`, `"uint64"`

  - Other: `"bool"`, `"complex64"`

  Not all functions support all types. See individual function
  documentation.

## Value

An mlx matrix with ones on the selected diagonal and zeros elsewhere.

## Details

MLX does not support `float64` operations on GPU. When this function
creates a `float64` array or converts one back to R, Rmlx temporarily
switches only that internal creation or layout work to CPU. Later
operations on the returned array still use the current
[`mlx_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_device.md).

## See also

[mlx.core.eye](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.eye)

## Examples

``` r
mlx_eye(3)
#> mlx array [3 x 3]
#>   dtype: float32
#>   values:
#>      [,1] [,2] [,3]
#> [1,]    1    0    0
#> [2,]    0    1    0
#> [3,]    0    0    1
mlx_eye(3, k = 1)
#> mlx array [3 x 3]
#>   dtype: float32
#>   values:
#>      [,1] [,2] [,3]
#> [1,]    0    1    0
#> [2,]    0    0    1
#> [3,]    0    0    0
```
