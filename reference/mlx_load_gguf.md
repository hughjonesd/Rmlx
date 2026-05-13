# Load MLX tensors from the GGUF format

Load MLX tensors from the GGUF format

## Usage

``` r
mlx_load_gguf(file)
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

  Named list where values are `NULL`, character vectors, or `mlx` arrays
  depending on the GGUF entry type.

## See also

<https://ml-explore.github.io/mlx/build/html/python/io.html#mlx.core.load_gguf>
