# Fill an mlx array with a constant value

Fill an mlx array with a constant value

## Usage

``` r
mlx_full(dim, value, dtype = NULL, device = mlx_default_device())
```

## Arguments

- dim:

  Integer vector specifying the array shape/dimensions.

- value:

  Scalar value used to fill the array. Numeric, logical, or complex.

- dtype:

  MLX dtype (`"float32"`, `"float64"`, `"bool"`, or `"complex64"`). If
  omitted, defaults to `"complex64"` for complex scalars, `"bool"` for
  logical scalars, and `"float32"` otherwise.

- device:

  Execution target: provide `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  Defaults to the current
  [`mlx_default_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_default_device.md).

## Value

An mlx array filled with the supplied value.

## See also

[mlx.core.full](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.full)

## Examples

``` r
filled <- mlx_full(c(2, 2), 3.14)
complex_full <- mlx_full(c(2, 2), 1+2i, dtype = "complex64")
```
