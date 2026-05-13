# Load MLX arrays from the safetensors format

Load MLX arrays from the safetensors format

## Usage

``` r
mlx_load_safetensors(file)
```

## Arguments

- file:

  Path to a `.npy` file. The extension is appended automatically when
  missing.

## Value

A list containing:

- `tensors`:

  Named list of `mlx` arrays.

- `metadata`:

  Named character vector with the serialized metadata.

## See also

<https://ml-explore.github.io/mlx/build/html/python/io.html#mlx.core.load_safetensors>
