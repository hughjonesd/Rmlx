# Softmax for mlx arrays

Softmax for mlx arrays

## Usage

``` r
mlx_softmax(x, axis = NULL, precise = FALSE)
```

## Arguments

- x:

  An mlx array, or an R array/matrix/vector that will be converted via
  [`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md).

- axis:

  Optional axis to operate over (1-indexed like R). When `NULL`, the
  array is flattened first.

- precise:

  Logical; compute in higher precision for stability.

## Value

An mlx array with normalized probabilities.

## See also

[mlx.core.softmax](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.softmax)

## Examples

``` r
x <- as_mlx(matrix(c(1, 2, 3, 4, 5, 6), 2, 3))
sm <- mlx_softmax(x, axis = 2)
rowSums(as.matrix(sm))
#> [1] 1 1
```
