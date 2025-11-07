# Reduce mlx arrays

These helpers mirror NumPy-style reductions, optionally collapsing one
or more axes. Use `drop = FALSE` to retain reduced axes with length one
(akin to setting `drop = FALSE` in base R).

## Usage

``` r
mlx_sum(x, axis = NULL, drop = TRUE)

mlx_prod(x, axis = NULL, drop = TRUE)

mlx_all(x, axis = NULL, drop = TRUE)

mlx_any(x, axis = NULL, drop = TRUE)

mlx_mean(x, axis = NULL, drop = TRUE)

mlx_var(x, axis = NULL, drop = TRUE, ddof = 0L)

mlx_std(x, axis = NULL, drop = TRUE, ddof = 0L)
```

## Arguments

- x:

  An mlx array, or an R array/matrix/vector that will be converted via
  [`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md).

- axis:

  Optional integer vector of axes (1-indexed) to reduce. When `NULL`,
  the reduction is performed over all elements.

- drop:

  Logical flag controlling dimension dropping: `TRUE` (default) removes
  reduced axes, while `FALSE` retains them with length one.

- ddof:

  Non-negative integer delta degrees of freedom for variance or standard
  deviation reductions.

## Value

An mlx array containing the reduction result.

## Details

`mlx_all()` and `mlx_any()` return mlx boolean scalars, while the base R
reducers [`all()`](https://rdrr.io/r/base/all.html) and
[`any()`](https://rdrr.io/r/base/any.html) applied to mlx inputs return
plain logical scalars.

## See also

[mlx.core.sum](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.sum),
[mlx.core.prod](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.prod),
[mlx.core.all](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.all),
[mlx.core.any](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.any),
[mlx.core.mean](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.mean),
[mlx.core.var](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.var),
[mlx.core.std](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.std)

## Examples

``` r
x <- as_mlx(matrix(1:4, 2, 2))
mlx_sum(x)
#> mlx array []
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 10
mlx_sum(x, axis = 1)
#> mlx array [2]
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 3 7
mlx_prod(x, axis = 2, drop = FALSE)
#> mlx array [2 x 1]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1]
#> [1,]    3
#> [2,]    8
mlx_all(x > 0)
#> mlx array []
#>   dtype: bool
#>   device: gpu
#>   values:
#> [1] TRUE
mlx_any(x > 3)
#> mlx array []
#>   dtype: bool
#>   device: gpu
#>   values:
#> [1] TRUE
mlx_mean(x, axis = 1)
#> mlx array [2]
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 1.5 3.5
mlx_var(x, axis = 2)
#> mlx array [2]
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 1 1
mlx_std(x, ddof = 1)
#> mlx array []
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 1.290994
```
