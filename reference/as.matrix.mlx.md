# Convert MLX array to R matrix/array

MLX arrays without dimension are returned as R vectors.

## Usage

``` r
# S3 method for class 'mlx'
as.matrix(x, ...)
```

## Arguments

- x:

  An mlx array.

- ...:

  Additional arguments (ignored)

## Value

A vector, matrix or array (numeric or logical depending on dtype)

## Examples

``` r
x <- as_mlx(matrix(1:4, 2, 2))
as.matrix(x)
#>      [,1] [,2]
#> [1,]    1    3
#> [2,]    2    4
```
