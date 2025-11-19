# Log cumulative sum exponential for mlx arrays

Log cumulative sum exponential for mlx arrays

## Usage

``` r
mlx_logcumsumexp(x, axis = NULL, reverse = FALSE, inclusive = TRUE)
```

## Arguments

- x:

  An mlx array, or an R array/matrix/vector that will be converted via
  [`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md).

- axis:

  Optional axis (single integer) to operate over.

- reverse:

  Logical flag for reverse accumulation.

- inclusive:

  Logical flag controlling inclusivity.

## Value

An mlx array.

## See also

[mlx.core.logaddexp](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.logaddexp)

## Examples

``` r
x <- as_mlx(1:4)
mlx_logcumsumexp(x)
#> mlx array [4]
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 1.000000 2.313262 3.407606 4.440190
m <- mlx_matrix(1:6, 2, 3)
mlx_logcumsumexp(m, axis = 2)
#> mlx array [2 x 3]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1]     [,2]     [,3]
#> [1,]    1 3.126928 5.142931
#> [2,]    2 4.126928 6.142931
```
