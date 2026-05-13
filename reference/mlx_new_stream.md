# MLX streams for asynchronous execution

Streams provide independent execution queues on a device, allowing
overlap of computation and finer control over scheduling.

`mlx_default_stream()` returns the current default stream for a device.

## Usage

``` r
mlx_new_stream(device = mlx_device())

mlx_default_stream(device = mlx_device())
```

## Arguments

- device:

  Execution target for APIs that expose a one-off device or stream
  override. Supply `"gpu"`, `"cpu"`, or an `mlx_stream` created via
  `mlx_new_stream()`. Ordinary array operations use the current
  [`mlx_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_device.md)
  instead.

## Value

An object of class `mlx_stream`.

## See also

<https://ml-explore.github.io/mlx/build/html/usage/using_streams.html>

## Examples

``` r
stream <- mlx_new_stream()
stream
#> mlx stream [cpu] index=2
```
