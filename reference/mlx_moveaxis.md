# Reorder mlx array axes

- `mlx_moveaxis()` repositions one or more axes to new locations.

- `aperm.mlx()` provides the familiar R interface, permuting axes
  according to `perm` via repeated calls to `mlx_moveaxis()`.

## Usage

``` r
mlx_moveaxis(x, source, destination)

# S3 method for class 'mlx'
aperm(a, perm = NULL, resize = TRUE, ...)
```

## Arguments

- x, a:

  An object coercible to mlx via
  [`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md).

- source:

  Integer vector of axis indices to move (1-indexed).

- destination:

  Integer vector giving the target positions for `source` axes
  (1-indexed). Must be the same length as `source`.

- perm:

  Integer permutation describing the desired axis order, matching the
  semantics of [`base::aperm()`](https://rdrr.io/r/base/aperm.html).

- resize:

  Logical flag from
  [`base::aperm()`](https://rdrr.io/r/base/aperm.html). Only `TRUE` is
  currently supported for mlx arrays.

- ...:

  Additional arguments accepted for compatibility; ignored.

## Value

An mlx array with axes permuted.

## See also

[mlx.core.moveaxis](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.moveaxis)

## Examples

``` r
x <- mlx_array(1:8, dim = c(2, 2, 2))
moved <- mlx_moveaxis(x, source = 1, destination = 3)
permuted <- aperm(x, c(2, 1, 3))
```
