# Convert MLX array to R matrix

MLX arrays with other than 2 dimensions are converted to a 1 column
matrix, with a warning.

## Usage

``` r
# S3 method for class 'mlx'
as.matrix(x, ...)
```

## Arguments

- x:

  An mlx array.

- ...:

  Additional arguments; ignored.

## Value

A vector, matrix or array (numeric or logical depending on dtype).

## Details

MLX does not support `float64` operations on GPU. When this function
creates a `float64` array or converts one back to R, Rmlx temporarily
switches only that internal creation or layout work to CPU. Later
operations on the returned array still use the current
[`mlx_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_device.md).

## Examples

``` r
x <- mlx_matrix(1:4, 2, 2)
as.matrix(x)
#>      [,1] [,2]
#> [1,]    1    3
#> [2,]    2    4
```
