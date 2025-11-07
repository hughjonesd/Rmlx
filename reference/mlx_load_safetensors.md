# Load MLX arrays from the safetensors format

Load MLX arrays from the safetensors format

## Usage

``` r
mlx_load_safetensors(file, device = mlx_default_device())
```

## Arguments

- file:

  Path to a `.npy` file. The extension is appended automatically when
  missing.

- device:

  Execution target: supply `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  Default:
  [`mlx_default_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_default_device.md).

## Value

A list containing:

- `tensors`:

  Named list of `mlx` arrays.

- `metadata`:

  Named character vector with the serialized metadata.

## See also

<https://ml-explore.github.io/mlx/build/html/python/io.html#mlx.core.load_safetensors>
