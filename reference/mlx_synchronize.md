# Synchronize MLX execution

Waits for outstanding operations on the specified device or stream to
complete.

## Usage

``` r
mlx_synchronize(device = mlx_device())
```

## Arguments

- device:

  Execution target for APIs that expose a one-off device or stream
  override. Supply `"gpu"`, `"cpu"`, or an `mlx_stream` created via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  Ordinary array operations use the current
  [`mlx_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_device.md)
  instead.

## Value

Returns `NULL` invisibly.

## See also

[mlx.core.default_device](https://ml-explore.github.io/mlx/build/html/python/metal.html)

## Examples

``` r
x <- mlx_matrix(1:4, 2, 2)
mlx_synchronize("cpu")
if (mlx_has_gpu()) mlx_synchronize("gpu")
stream <- mlx_new_stream()
mlx_synchronize(stream)
```
