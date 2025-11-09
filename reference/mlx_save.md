# Save an MLX array to disk

Persists an MLX array to a `.npy` file using MLX's native serialization.

## Usage

``` r
mlx_save(x, file)
```

## Arguments

- x:

  Object coercible to `mlx`.

- file:

  Path to the output file. If the file does not end with `.npy`, the
  extension is appended automatically.

## Value

Invisibly returns the full path that was written, including the `.npy`
suffix.

## See also

<https://ml-explore.github.io/mlx/build/html/python/io.html#mlx.core.save>

## Examples

``` r
path <- tempfile(fileext = ".mlx")
mlx_save(as_mlx(matrix(1:4, 2, 2), device = "cpu"), path)
restored <- mlx_load(path, device = "cpu")
```
