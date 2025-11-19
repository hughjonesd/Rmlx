# Split mlx arrays along a margin

`asplit()` extends base `asplit()` to work with mlx arrays by delegating
to
[`mlx_split()`](https://hughjonesd.github.io/Rmlx/reference/mlx_pad.md).
When `x` is_mlx the result is a list of mlx arrays; otherwise, the base
implementation is used.

## Usage

``` r
asplit(x, MARGIN, drop = FALSE)

# Default S3 method
asplit(x, MARGIN, drop = FALSE)

# S3 method for class 'mlx'
asplit(x, MARGIN, drop = FALSE)
```

## Arguments

- x:

  an array, including a matrix.

- MARGIN:

  a vector giving the margins to split by. E.g., for a matrix `1`
  indicates rows, `2` indicates columns, `c(1, 2)` indicates rows and
  columns. Where `x` has named dimnames, it can be a character vector
  selecting dimension names.

- drop:

  a logical indicating whether the splits should drop dimensions and
  dimnames.

## Value

For mlx inputs, a list of mlx arrays; otherwise matches
[`base::asplit()`](https://rdrr.io/r/base/asplit.html).

## Details

Currently only a single `MARGIN` value is supported for mlx arrays.
