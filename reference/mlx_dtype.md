# Get the data type of an MLX array

Get the data type of an MLX array

## Usage

``` r
mlx_dtype(x)
```

## Arguments

- x:

  An mlx array, or an R array/matrix/vector that will be converted via
  [`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md).

## Value

A data type string (see
[`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md) for
possibilities).

## Examples

``` r
x <- as_mlx(matrix(1:6, 2, 3))
mlx_dtype(x)
#> [1] "float32"
```
