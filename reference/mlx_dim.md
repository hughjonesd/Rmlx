# Get dimensions helper

Get dimensions helper

## Usage

``` r
mlx_dim(x)
```

## Arguments

- x:

  An mlx array, or an R array/matrix/vector that will be converted via
  [`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md).

## Value

Dimensions

## Examples

``` r
x <- as_mlx(matrix(1:6, 2, 3))
mlx_dim(x)
#> [1] 2 3
```
