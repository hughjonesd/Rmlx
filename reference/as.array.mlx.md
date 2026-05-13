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

## Details

MLX does not support `float64` operations on GPU. When this function
creates a `float64` array or converts one back to R, Rmlx temporarily
switches only that internal creation or layout work to CPU. Later
operations on the returned array still use the current
[`mlx_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_device.md).

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
