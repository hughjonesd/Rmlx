# Take values using per-position axis indices

Mirrors
[`mlx.core.take_along_axis()`](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.take_along_axis.html)
while accepting 1-based R indices.

## Usage

``` r
mlx_take_along_axis(x, indices, axis)
```

## Arguments

- x:

  An mlx array.

- indices:

  Integer positions along `axis`. Must be broadcast-compatible with `x`
  except at the selected axis.

- axis:

  Axis to index (1-based).

## Value

An `mlx` array. Names on the indexed axis are dropped because
per-position indices may reorder each slice differently.

## Details

If `y <- mlx_take_along_axis(x, idx, axis)` where `x` is an `m x n`
matrix and `idx` is a matrix:

- `y` will have the same shape as `idx`, possibly after `idx` has been
  broadcast to the dimensions of `y` for all axes except `axis`.

- For `axis = 1`, values of `idx` give the row, and columns are in
  order: `y[i, j]` equals `x[idx[i, j], j]`. `idx` must have 1 or `n`
  columns. `y` will have the same number of rows as `idx`.

- For `axis = 2`, values of `idx` give the column, and rows are in
  order: `y[i, j]` equals `x[i, idx[i, j]]`. `idx` must have 1 or `m`
  rows, and `y` will have the same number of columns as `idx`.

More generally, for `x` and `idx` of `d` dimensions, and `axis = a`:

- `y[i_1, ...., i_d]` equals `x[i_1, ..., idx[i_1,...,i_d], ..., i_d]`
  where the `idx` vector is in position `a`.

For broadcasting, the simplest rule is that if `idx` has 1 column,
`mlx_take_along_axis(x, idx, 1)` is the same as `x[drop(idx),]`; and if
`idx` has 1 row, `mlx_take_along_axis(x, idx, 2)` is the same as
`x[, drop(idx)]`.

## Examples

``` r
x <- outer(1:3, c(0.1, 0.2), "+")
x <- as_mlx(x)
x
#> mlx array [3 x 2]
#>   dtype: float32
#>   values:
#>      [,1] [,2]
#> [1,]  1.1  1.2
#> [2,]  2.1  2.2
#> [3,]  3.1  3.2

idx_cols <- matrix(c(1, 2,
                     2, 2,
                     1, 1), nrow = 3, byrow = TRUE)
mlx_take_along_axis(x, idx_cols, axis = 2)
#> mlx array [3 x 2]
#>   dtype: float32
#>   values:
#>      [,1] [,2]
#> [1,]  1.1  1.2
#> [2,]  2.2  2.2
#> [3,]  3.1  3.1

idx_rows <- matrix(c(1, 2,
                     3, 1), nrow = 2, byrow = TRUE)
mlx_take_along_axis(x, idx_rows, axis = 1)
#> mlx array [2 x 2]
#>   dtype: float32
#>   values:
#>      [,1] [,2]
#> [1,]  1.1  2.2
#> [2,]  3.1  1.2
```
