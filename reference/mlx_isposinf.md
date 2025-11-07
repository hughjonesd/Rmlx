# Detect signed infinities in mlx arrays

`mlx_isposinf()` and `mlx_isneginf()` mirror
[`mlx.core.isposinf()`](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.isposinf)
and
[`mlx.core.isneginf()`](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.isneginf),
returning boolean masks of positive or negative infinities.

## Usage

``` r
mlx_isposinf(x)

mlx_isneginf(x)
```

## Arguments

- x:

  An mlx array.

## Value

An mlx boolean array highlighting infinite entries.

## See also

[mlx.core.isposinf](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.isposinf),
[mlx.core.isneginf](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.isneginf)

## Examples

``` r
vals <- as_mlx(c(-Inf, -1, 0, Inf))
as.matrix(mlx_isposinf(vals))
#> [1] FALSE FALSE FALSE  TRUE
as.matrix(mlx_isneginf(vals))
#> [1]  TRUE FALSE FALSE FALSE
```
