# Load an MLX array from disk

Restores an array saved with
[`mlx_save()`](https://hughjonesd.github.io/Rmlx/reference/mlx_save.md).

## Usage

``` r
mlx_load(file)
```

## Arguments

- file:

  Path to a `.npy` file. The extension is appended automatically when
  missing.

## Value

An `mlx` array containing the file contents.

## See also

<https://ml-explore.github.io/mlx/build/html/python/io.html#mlx.core.load>
