# Convert MLX array to logical vector

Convert MLX array to logical vector

## Usage

``` r
# S3 method for class 'mlx'
as.logical(x, ...)
```

## Arguments

- x:

  An mlx array.

- ...:

  Additional arguments passed to
  [`base::as.vector()`](https://rdrr.io/r/base/vector.html).

## Value

A logical vector.

## Examples

``` r
x <- as_mlx(c(1, 0, 2))
as.logical(x)
#> [1]  TRUE FALSE  TRUE
```
