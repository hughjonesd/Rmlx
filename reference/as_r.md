# Convert MLX array to base R objects

`as_r()` mirrors base R coercion rules: MLX objects with
[`dim()`](https://rdrr.io/r/base/dim.html) equal to `NULL` return a
plain vector, while higher-dimensional inputs return matrices or arrays.

## Usage

``` r
as_r(x, ...)
```

## Arguments

- x:

  An mlx array.

- ...:

  Additional arguments; ignored.

## Value

A vector, matrix, or array depending on the dimensions of `x`.

## See also

[`as.array.mlx()`](https://hughjonesd.github.io/Rmlx/reference/as.array.mlx.md),
[`as.vector.mlx()`](https://hughjonesd.github.io/Rmlx/reference/as.vector.mlx.md),
[`as.matrix.mlx()`](https://hughjonesd.github.io/Rmlx/reference/as.matrix.mlx.md)

## Examples

``` r
v <- as_mlx(1:3)
as_r(v)      # numeric vector
#> [1] 1 2 3
```
