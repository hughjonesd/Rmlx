# Get length of MLX array

Get length of MLX array

## Usage

``` r
# S3 method for class 'mlx'
length(x)
```

## Arguments

- x:

  An mlx array, or an R array/matrix/vector that will be converted via
  [`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md).

## Value

Total number of elements.

## Examples

``` r
x <- mlx_matrix(1:6, 2, 3)
length(x)
#> [1] 6
```
