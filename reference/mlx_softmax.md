# Softmax for mlx arrays

Softmax for mlx arrays

## Usage

``` r
mlx_softmax(x, axes = NULL, precise = FALSE)
```

## Arguments

- x:

  An mlx array, or an R array/matrix/vector that will be converted via
  [`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md).

- axes:

  Integer vector of axes (1-indexed). Supply positive integers between 1
  and the array rank. Many helpers interpret `NULL` to mean "all
  axes"—see the function details for specifics.

- precise:

  Logical; compute in higher precision for stability.

## Value

An mlx array with normalized probabilities.

## See also

[mlx.core.softmax](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.softmax)

## Examples

``` r
x <- mlx_matrix(1:6, 2, 3)
sm <- mlx_softmax(x, axes = 2)
rowSums(sm)
#> mlx array [2]
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 1 1
```
