# Get device associated with an MLX object

Get device associated with an MLX object

## Usage

``` r
mlx_device(x)
```

## Arguments

- x:

  An mlx array, or an R array/matrix/vector that will be converted via
  [`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md).

## Value

`"gpu"` or `"cpu"`.

## Examples

``` r
x <- as_mlx(1:10)
mlx_device(x)
#> [1] "gpu"
```
