# Get data type helper

Get data type helper

## Usage

``` r
mlx_dtype(x)
```

## Arguments

- x:

  An mlx array, or an R array/matrix/vector that will be converted via
  [`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md).

## Value

Data type string

## Examples

``` r
x <- as_mlx(matrix(1:6, 2, 3))
mlx_dtype(x)
#> [1] "float32"
```
