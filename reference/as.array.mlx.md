# Convert MLX array to R array

Always returns an R array using the MLX shape. One-dimensional MLX
inputs become 1-D arrays (with `dim` set to their length) instead of
plain vectors.

## Usage

``` r
# S3 method for class 'mlx'
as.array(x, ...)
```

## Arguments

- x:

  An mlx array.

- ...:

  Additional arguments; ignored.

## Value

An R array with the same shape as the MLX input.

## See also

[`as_r()`](https://hughjonesd.github.io/Rmlx/reference/as_r.md),
[`as.vector.mlx()`](https://hughjonesd.github.io/Rmlx/reference/as.vector.mlx.md),
[`as.matrix.mlx()`](https://hughjonesd.github.io/Rmlx/reference/as.matrix.mlx.md)

## Examples

``` r
x <- mlx_matrix(1:8, 2, 4)
as.array(x)
#>      [,1] [,2] [,3] [,4]
#> [1,]    1    3    5    7
#> [2,]    2    4    6    8

v <- as_mlx(1:3)
as.array(v)  # 1-D array with dim 3
#> [1] 1 2 3
```
