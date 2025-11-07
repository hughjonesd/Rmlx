# Save MLX arrays to the GGUF format

Save MLX arrays to the GGUF format

## Usage

``` r
mlx_save_gguf(file, arrays, metadata = list())
```

## Arguments

- file:

  Output path. `.safetensors` is appended automatically when omitted.

- arrays:

  Named list of objects coercible to `mlx`.

- metadata:

  Optional named list describing GGUF metadata. Values may be character
  vectors or `mlx` arrays.

## Value

Invisibly returns the full path that was written.

## See also

<https://ml-explore.github.io/mlx/build/html/python/io.html#mlx.core.save_gguf>
