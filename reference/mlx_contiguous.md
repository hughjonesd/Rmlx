# Ensure contiguous memory layout

Returns a copy of `x` with contiguous strides on the requested device or
stream.

## Usage

``` r
mlx_contiguous(x, device = NULL)
```

## Arguments

- x:

  An mlx array.

- device:

  Execution target: supply `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  Defaults to the current
  [`mlx_default_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_default_device.md)
  unless noted otherwise (helpers that act on an existing array
  typically reuse that array's device or stream).

## Value

An mlx array backed by contiguous storage on the specified device.

## See also

<https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.contiguous>

## Examples

``` r
x <- mlx_swapaxes(as_mlx(matrix(1:4, 2, 2)), axis1 = 1, axis2 = 2)
y <- mlx_contiguous(x)
identical(as.array(x), as.array(y))
#> [1] TRUE
```
