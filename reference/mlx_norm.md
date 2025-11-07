# Matrix and vector norms for mlx arrays

Matrix and vector norms for mlx arrays

## Usage

``` r
mlx_norm(x, ord = NULL, axis = NULL, drop = TRUE)
```

## Arguments

- x:

  An mlx array.

- ord:

  Numeric or character norm order. Use `NULL` for the default 2-norm.

- axis:

  Optional integer vector of axes (1-indexed) along which to compute the
  norm.

- drop:

  If `TRUE` (default), drop dimensions of length 1. If `FALSE`, retain
  all dimensions. Equivalent to `keepdims = TRUE` in underlying mlx
  functions.

## Value

An mlx array containing the requested norm.

## See also

[mlx.linalg.norm](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.norm)

## Examples

``` r
x <- as_mlx(matrix(1:4, 2, 2))
mlx_norm(x)
#> mlx array []
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 5.477226
mlx_norm(x, ord = 2)
#> mlx array []
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 5.464986
mlx_norm(x, axis = 2)
#> mlx array [2]
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 3.162278 4.472136
```
