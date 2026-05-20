# Scale mlx arrays

Extends base [`scale()`](https://rdrr.io/r/base/scale.html) to handle
mlx inputs with MLX reductions and broadcasting. Computed center and
scale attributes are mlx arrays. User-supplied attributes keep their
input type: mlx inputs stay mlx, while base R vectors stay base R
vectors.

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
