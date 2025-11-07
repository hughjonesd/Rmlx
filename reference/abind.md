# Bind mlx arrays along an axis

Bind mlx arrays along an axis

## Usage

``` r
abind(..., along = 1L)
```

## Arguments

- ...:

  One or more mlx arrays (or a single list of arrays) to combine.

- along:

  Positive integer giving the existing axis (1-indexed) along which to
  bind the inputs.

## Value

An mlx array formed by concatenating the inputs along `along`.

## Details

This is an MLX-backed alternative to `abind::abind()`. All inputs must
share the same shape on non-bound axes. The `along` axis must already
exist; to create a new axis use
[`mlx_stack()`](https://hughjonesd.github.io/Rmlx/reference/mlx_stack.md).

## See also

[mlx.core.concatenate](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.concatenate)

## Examples

``` r
x <- as_mlx(array(1:12, c(2, 3, 2)))
y <- as_mlx(array(13:24, c(2, 3, 2)))
z <- abind(x, y, along = 3)
dim(z)
#> [1] 2 3 4
```
