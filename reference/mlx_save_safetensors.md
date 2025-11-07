# Save MLX arrays to the safetensors format

Save MLX arrays to the safetensors format

## Usage

``` r
mlx_save_safetensors(file, arrays, metadata = character())
```

## Arguments

- file:

  Output path. `.safetensors` is appended automatically when omitted.

- arrays:

  Named list of objects coercible to `mlx`.

- metadata:

  Optional named character vector of metadata entries.

## Value

Invisibly returns the full path that was written.

## See also

<https://ml-explore.github.io/mlx/build/html/python/io.html#mlx.core.save_safetensors>
