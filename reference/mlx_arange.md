# Numerical ranges on MLX devices

`mlx_arange()` creates evenly spaced values starting at `start`,
stepping by `step`, up to and including `stop` (if exactly reachable).
This matches R's [`base::seq()`](https://rdrr.io/r/base/seq.html)
behavior.

## Usage

``` r
mlx_arange(
  start,
  stop,
  step = 1,
  dtype = c("float32", "float64", "int8", "int16", "int32", "int64", "uint8", "uint16",
    "uint32", "uint64")
)
```

## Arguments

- start:

  Starting value.

- stop:

  Upper bound (included if exactly reachable by the step sequence).

- step:

  Step size (defaults to 1).

- dtype:

  Data type string. Supported types include:

  - Floating point: `"float32"`, `"float64"`

  - Integer: `"int8"`, `"int16"`, `"int32"`, `"int64"`, `"uint8"`,
    `"uint16"`, `"uint32"`, `"uint64"`

  - Other: `"bool"`, `"complex64"`

  Not all functions support all types. See individual function
  documentation.

## Value

A 1D mlx array.

## Details

MLX does not support `float64` operations on GPU. When this function
creates a `float64` array or converts one back to R, Rmlx temporarily
switches only that internal creation or layout work to CPU. Later
operations on the returned array still use the current
[`mlx_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_device.md).

## Difference from Python/C++

Unlike Python's [`range()`](https://rdrr.io/r/base/range.html) and
`numpy.arange()` which use an exclusive upper bound, `mlx_arange()`
matches R's [`base::seq()`](https://rdrr.io/r/base/seq.html) by
including `stop` only if it's exactly reachable by the step sequence.
This is consistent with
[`mlx_linspace()`](https://hughjonesd.github.io/Rmlx/reference/mlx_linspace.md)
and
[`mlx_slice_update()`](https://hughjonesd.github.io/Rmlx/reference/mlx_slice_update.md),
which also follow R conventions.

## See also

[mlx.core.arange](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.arange)

## Examples

``` r
mlx_arange(0, 4)        # 0, 1, 2, 3, 4
#> mlx array [5]
#>   dtype: float32
#>   values:
#> [1] 0 1 2 3 4
mlx_arange(1, 5)        # 1, 2, 3, 4, 5
#> mlx array [5]
#>   dtype: float32
#>   values:
#> [1] 1 2 3 4 5
mlx_arange(1, 9, 2)     # 1, 3, 5, 7, 9
#> mlx array [5]
#>   dtype: float32
#>   values:
#> [1] 1 3 5 7 9
mlx_arange(1, 6, 2)     # 1, 3, 5 (6 not reachable)
#> mlx array [3]
#>   dtype: float32
#>   values:
#> [1] 1 3 5
```
