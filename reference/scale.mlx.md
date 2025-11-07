# Scale mlx arrays

Extends base [`scale()`](https://rdrr.io/r/base/scale.html) to handle
mlx inputs without moving data back to base R. The computation delegates
to MLX reductions and broadcasting. When centering or scaling values are
computed, the attributes `"scaled:center"` and `"scaled:scale"` are
stored as 1 x `ncol(x)` mlx arrays (user-supplied numeric vectors are
preserved as-is). These attributes remain MLX arrays even after coercing
with [`as.matrix()`](https://rdrr.io/r/base/matrix.html), so they stay
lazily evaluated.

## Usage

``` r
# S3 method for class 'mlx'
scale(x, center = TRUE, scale = TRUE)
```

## Arguments

- x:

  a numeric matrix(like object).

- center:

  either a logical value or numeric-alike vector of length equal to the
  number of columns of `x`, where ‘numeric-alike’ means that
  [`as.numeric`](https://rdrr.io/r/base/numeric.html)`(.)` will be
  applied successfully if
  [`is.numeric`](https://rdrr.io/r/base/numeric.html)`(.)` is not true.

- scale:

  either a logical value or a numeric-alike vector of length equal to
  the number of columns of `x`.

## Value

An mlx array with centred/scaled columns.
