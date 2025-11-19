# Test if object is an MLX array

Test if object is an MLX array

## Usage

``` r
is_mlx(x)
```

## Arguments

- x:

  Object to test

## Value

Logical scalar.

## Examples

``` r
x <- as_mlx(matrix(1:4, 2, 2))
is_mlx(x)
#> [1] TRUE
```
