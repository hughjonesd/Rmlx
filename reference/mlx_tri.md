# Triangular helpers for MLX arrays

`mlx_tri()` creates a lower-triangular mask (ones on and below a
diagonal, zeros elsewhere). `mlx_tril()` and `mlx_triu()` retain only
the lower or upper triangular part of an existing array, respectively.

## Usage

``` r
mlx_tri(
  n,
  m = NULL,
  k = 0L,
  dtype = c("float32", "float64"),
  device = mlx_default_device()
)

mlx_tril(x, k = 0L)

mlx_triu(x, k = 0L)
```

## Arguments

- n:

  Number of rows.

- m:

  Optional number of columns (defaults to `n` for square output).

- k:

  Diagonal offset: `0` selects the main diagonal, positive values move
  to the upper diagonals, negative values to the lower diagonals.

- dtype:

  Data type string. Supported types include:

  - Floating point: `"float32"`, `"float64"`

  - Integer: `"int8"`, `"int16"`, `"int32"`, `"int64"`, `"uint8"`,
    `"uint16"`, `"uint32"`, `"uint64"`

  - Other: `"bool"`, `"complex64"`

  `float64` arrays are CPU-only. Use `device = "cpu"` when creating or
  casting to `float64`, and cast back to `float32` before using the GPU.
  Not all functions support all types. See individual function
  documentation.

- device:

  Execution target: supply `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  By default, many functions use the
  [`mlx_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_device.md)
  of their first argument.

- x:

  Object coercible to `mlx`.

## Value

An `mlx` array.

## See also

[mlx.core.tri](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.tri)

## Examples

``` r
mlx_tri(3)          # 3x3 lower-triangular mask
#> mlx array [3 x 3]
#>   dtype: float32
#>   device: cpu
#>   values:
#>      [,1] [,2] [,3]
#> [1,]    1    0    0
#> [2,]    1    1    0
#> [3,]    1    1    1
mlx_tril(diag(3) + 2)  # keep lower part of a matrix
#> mlx array [3 x 3]
#>   dtype: float32
#>   device: cpu
#>   values:
#>      [,1] [,2] [,3]
#> [1,]    3    0    0
#> [2,]    2    3    0
#> [3,]    2    2    3
```
