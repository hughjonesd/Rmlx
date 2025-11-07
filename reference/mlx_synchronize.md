# Synchronize MLX execution

Waits for outstanding operations on the specified device or stream to
complete.

## Usage

``` r
mlx_synchronize(device = c("gpu", "cpu"))
```

## Arguments

- device:

  Device identifier ("gpu" or "cpu") or an `mlx_stream` created by
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).

## See also

[mlx.core.default_device](https://ml-explore.github.io/mlx/build/html/python/metal.html)

## Examples

``` r
x <- as_mlx(matrix(1:4, 2, 2))
mlx_synchronize("gpu")
stream <- mlx_new_stream()
mlx_synchronize(stream)
```
