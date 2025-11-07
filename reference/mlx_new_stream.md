# MLX streams for asynchronous execution

Streams provide independent execution queues on a device, allowing
overlap of computation and finer control over scheduling.

`mlx_default_stream()` returns the current default stream for a device.

## Usage

``` r
mlx_new_stream(device = mlx_default_device())

mlx_default_stream(device = mlx_default_device())
```

## Arguments

- device:

  Device identifier (`"gpu"` or `"cpu"`). Defaults to the current
  [`mlx_default_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_default_device.md).

## Value

An object of class `mlx_stream`.

## See also

<https://ml-explore.github.io/mlx/build/html/usage/using_streams.html>

## Examples

``` r
stream <- mlx_new_stream()
stream
#> mlx stream [gpu] index=2
```
