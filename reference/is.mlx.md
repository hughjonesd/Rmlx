# Test if object is an MLX array

Test if object is an MLX array

## Usage

``` r
is.mlx(x)
```

## Arguments

- x:

  Object to test

## Value

Logical

## Examples

``` r
x <- as_mlx(matrix(1:4, 2, 2))
is.mlx(x)
#> [1] TRUE
```
