# Convert MLX array to R array

Convert MLX array to R array

## Usage

``` r
# S3 method for class 'mlx'
as.array(x, ...)
```

## Arguments

- x:

  An mlx array.

- ...:

  Additional arguments (ignored)

## Value

A numeric array

## Examples

``` r
x <- as_mlx(matrix(1:8, 2, 4))
as.array(x)
#>      [,1] [,2] [,3] [,4]
#> [1,]    1    3    5    7
#> [2,]    2    4    6    8
```
