# Argmax and argmin on mlx arrays

Argmax and argmin on mlx arrays

## Usage

``` r
mlx_argmax(x, axis = NULL, drop = TRUE)

mlx_argmin(x, axis = NULL, drop = TRUE)
```

## Arguments

- x:

  An mlx array, or an R array/matrix/vector that will be converted via
  [`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md).

- axis:

  Single axis (1-indexed). Supply a positive integer between 1 and the
  array rank. Use `NULL` when the helper interprets it as "all axes"
  (see individual docs).

- drop:

  If `TRUE` (default), drop dimensions of length 1. If `FALSE`, retain
  all dimensions. Equivalent to `keepdims = TRUE` in underlying mlx
  functions.

## Value

An mlx array of indices. Indices are 1-based to match R's conventions.

## Details

When `axis = NULL`, the array is flattened before computing extrema.
Setting `drop = FALSE` retains the reduced axis as length one in the
returned indices.

## See also

[mlx.core.argmax](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.argmax),
[mlx.core.argmin](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.argmin)

## Examples

``` r
x <- as_mlx(matrix(c(1, 5, 3, 2), 2, 2))
mlx_argmax(x)
#> mlx array []
#>   dtype: int64
#>   device: gpu
#>   values:
#> [1] 3
mlx_argmax(x, axis = 1)
#> mlx array []
#>   dtype: int64
#>   device: gpu
#>   values:
#> [1] 2 1
mlx_argmin(x)
#> mlx array []
#>   dtype: int64
#>   device: gpu
#>   values:
#> [1] 1
```
