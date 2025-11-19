# Log-sum-exp reduction for mlx arrays

Log-sum-exp reduction for mlx arrays

## Usage

``` r
mlx_logsumexp(x, axes = NULL, drop = TRUE)
```

## Arguments

- x:

  An mlx array, or an R array/matrix/vector that will be converted via
  [`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md).

- axes:

  Integer vector of axes (1-indexed). Supply positive integers between 1
  and the array rank. Many helpers interpret `NULL` to mean "all
  axes"—see the function details for specifics.

- drop:

  Logical indicating whether the reduced axes should be dropped (default
  `TRUE`).

## Value

An mlx array containing log-sum-exp results.

## See also

[mlx.core.logsumexp](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.logsumexp)

## Examples

``` r
x <- mlx_matrix(1:6, 2, 3)
mlx_logsumexp(x)
#> mlx array []
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 6.456193
mlx_logsumexp(x, axes = 2)
#> mlx array [2]
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 5.142931 6.142931
```
