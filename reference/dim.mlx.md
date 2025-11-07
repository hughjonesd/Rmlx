# Get dimensions of MLX array

Get dimensions of MLX array

## Usage

``` r
# S3 method for class 'mlx'
dim(x)
```

## Arguments

- x:

  An mlx array, or an R array/matrix/vector that will be converted via
  [`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md).

## Value

Integer vector of dimensions

## Examples

``` r
x <- as_mlx(matrix(1:4, 2, 2))
dim(x)
#> [1] 2 2
```
