# Set the default MLX stream

Set the default MLX stream

## Usage

``` r
mlx_set_default_stream(stream)
```

## Arguments

- stream:

  An object created by
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md)
  or
  [`mlx_default_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).

## Value

Invisibly returns `stream`.

## Examples

``` r
stream <- mlx_new_stream()
old <- mlx_default_stream()
mlx_set_default_stream(stream)
mlx_set_default_stream(old)  # restore
```
