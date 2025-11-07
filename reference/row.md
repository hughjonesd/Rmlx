# Row and column indices for mlx arrays

Extends base `row()` and `col()` so they also accept mlx arrays. When
`as.factor = FALSE` the result stays on the MLX backend, avoiding
round-tripping through base R.

## Usage

``` r
row(x, as.factor = FALSE)

# Default S3 method
row(x, as.factor = FALSE)

# S3 method for class 'mlx'
row(x, as.factor = FALSE)

col(x, as.factor = FALSE)

# Default S3 method
col(x, as.factor = FALSE)

# S3 method for class 'mlx'
col(x, as.factor = FALSE)
```

## Arguments

- x:

  a matrix-like object, that is one with a two-dimensional `dim`.

- as.factor:

  a logical value indicating whether the value should be returned as a
  factor of row labels (created if necessary) rather than as numbers.

## Value

A matrix or array of row indices (for `row()`) or column indices (for
`col()`), matching the base R behaviour.
