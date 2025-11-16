# Load an MLX array from disk

Restores an array saved with
[`mlx_save()`](https://hughjonesd.github.io/Rmlx/reference/mlx_save.md)
and optionally places it on a specified device.

## Usage

``` r
mlx_load(file, device = mlx_default_device())
```

## Arguments

- file:

  Path to a `.npy` file. The extension is appended automatically when
  missing.

- device:

  Execution target: supply `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  Defaults to the current
  [`mlx_default_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_default_device.md)
  unless noted otherwise (helpers that act on an existing array
  typically reuse that array's device or stream).

## Value

An `mlx` array containing the file contents.

## Details

Use an `mlx_stream` from
[`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md)
to load directly onto a specific stream; otherwise the array is placed
on the current
[`mlx_default_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_default_device.md).

## See also

<https://ml-explore.github.io/mlx/build/html/python/io.html#mlx.core.load>
