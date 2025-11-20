# Cumulative sum and product

Compute cumulative sums or products along an axis.

## Usage

``` r
mlx_cumsum(x, axis = NULL, reverse = FALSE, inclusive = TRUE)

mlx_cumprod(x, axis = NULL, reverse = FALSE, inclusive = TRUE)
```

## Arguments

- x:

  An mlx array, or an R array/matrix/vector that will be converted via
  [`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md).

- axis:

  Single axis (1-indexed). Supply a positive integer between 1 and the
  array rank. Use `NULL` when the helper interprets it as "all axes"
  (see individual docs).

- reverse:

  If `TRUE`, compute in reverse order.

- inclusive:

  If `TRUE` (default), include the current element in the cumulative
  operation. If `FALSE`, the cumulative operation is exclusive (starts
  from identity element).

## Value

An mlx array with cumulative sums or products.

## Details

When `axis` is `NULL` (default), the array is flattened before computing
the cumulative result.

## See also

[`cumsum()`](https://rdrr.io/r/base/cumsum.html),
[`cumprod()`](https://rdrr.io/r/base/cumsum.html),
[mlx.core.cumsum](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.cumsum),
[mlx.core.cumprod](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.cumprod)

## Examples

``` r
x <- as_mlx(1:5)
mlx_cumsum(x)  # [1, 3, 6, 10, 15]
#> mlx array [5]
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1]  1  3  6 10 15

mat <- mlx_matrix(1:12, 3, 4)
mlx_cumsum(mat, axis = 1)  # cumsum down rows
#> mlx array [3 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1] [,2] [,3] [,4]
#> [1,]    1    4    7   10
#> [2,]    3    9   15   21
#> [3,]    6   15   24   33
```
