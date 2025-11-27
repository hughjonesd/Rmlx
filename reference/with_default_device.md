# Temporarily set the default MLX device or stream

Use `local_default_device()` to temporarily switch devices within the
current function.

## Usage

``` r
with_default_device(device, code)

local_default_device(device, .local_envir = parent.frame())
```

## Arguments

- device:

  `"gpu"`, `"cpu"`, or an `mlx_stream` created via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).

- code:

  Expression to evaluate while `device` is active.

- .local_envir:

  Environment to bind the restoration to. Defaults to the calling
  environment.

## Value

The result of evaluating `code`.

Invisibly returns the previous default device.

## See also

[mlx.core.default_device](https://ml-explore.github.io/mlx/build/html/python/metal.html)

## Examples

``` r
with_default_device("cpu", x <- mlx_vector(1:10))

local_default_device("cpu")
# code here runs on CPU, then the previous default is restored
```
