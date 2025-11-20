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

  Additional arguments (ignored)

## Value

A vector, matrix or array (numeric or logical depending on dtype).

## Examples

``` r
x <- mlx_matrix(1:4, 2, 2)
as.matrix(x)
#>      [,1] [,2]
#> [1,]    1    3
#> [2,]    2    4
```
