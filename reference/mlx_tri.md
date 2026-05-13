# Triangular helpers for MLX arrays

`mlx_tri()` creates a lower-triangular mask (ones on and below a
diagonal, zeros elsewhere). `mlx_tril()` and `mlx_triu()` retain only
the lower or upper triangular part of an existing array, respectively.

## Usage

``` r
mlx_tri(n, m = NULL, k = 0L, dtype = c("float32", "float64"))

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

  Not all functions support all types. See individual function
  documentation.

- x:

  Object coercible to `mlx`.

## Value

An `mlx` array.

## Details

MLX does not support `float64` operations on GPU. When this function
creates a `float64` array or converts one back to R, Rmlx temporarily
switches only that internal creation or layout work to CPU. Later
operations on the returned array still use the current
[`mlx_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_device.md).

## See also

[mlx.core.tri](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.tri)

## Examples

``` r
mlx_tri(3)          # 3x3 lower-triangular mask
#> mlx array [3 x 3]
#>   dtype: float32
#>   values:
#>      [,1] [,2] [,3]
#> [1,]    1    0    0
#> [2,]    1    1    0
#> [3,]    1    1    1
mlx_tril(diag(3) + 2)  # keep lower part of a matrix
#> mlx array [3 x 3]
#>   dtype: float32
#>   values:
#>      [,1] [,2] [,3]
#> [1,]    3    0    0
#> [2,]    2    3    0
#> [3,]    2    2    3
```
