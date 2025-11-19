# Synchronize MLX execution

Waits for outstanding operations on the specified device or stream to
complete.

## Usage

``` r
mlx_synchronize(device = mlx_default_device())
```

## Arguments

- device:

  Execution target: supply `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  Defaults to the current
  [`mlx_default_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_default_device.md)
  unless noted otherwise (helpers that act on an existing array
  typically reuse that array's device or stream).

## Value

Returns `NULL` invisibly.

## See also

[mlx.core.default_device](https://ml-explore.github.io/mlx/build/html/python/metal.html)

## Examples

``` r
x <- as_mlx(matrix(1:4, 2, 2))
mlx_synchronize("gpu")
stream <- mlx_new_stream()
mlx_synchronize(stream)
```
