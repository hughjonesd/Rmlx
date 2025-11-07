# Numerical ranges on MLX devices

`mlx_arange()` mirrors [`base::seq()`](https://rdrr.io/r/base/seq.html)
with mlx arrays: it creates evenly spaced values starting at `start`
(default `0`), stepping by `step` (default `1`), and stopping before
`stop`.

## Usage

``` r
mlx_arange(
  stop,
  start = NULL,
  step = NULL,
  dtype = c("float32", "float64", "int8", "int16", "int32", "int64", "uint8", "uint16",
    "uint32", "uint64"),
  device = mlx_default_device()
)
```

## Arguments

- stop:

  Exclusive upper bound.

- start:

  Optional starting value (defaults to 0).

- step:

  Optional step size (defaults to 1).

- dtype:

  MLX dtype (`"float32"` or `"float64"`).

- device:

  Execution target: provide `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  Defaults to the current
  [`mlx_default_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_default_device.md).

## Value

A 1D mlx array.

## See also

[mlx.core.arange](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.arange)

## Examples

``` r
mlx_arange(5)                    # 0, 1, 2, 3, 4
#> mlx array [5]
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 0 1 2 3 4
mlx_arange(5, start = 1, step = 2) # 1, 3
#> mlx array [2]
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 1 3
```
