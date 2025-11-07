# Arithmetic and comparison operators for MLX arrays

Arithmetic and comparison operators for MLX arrays

## Usage

``` r
# S3 method for class 'mlx'
Ops(e1, e2 = NULL)
```

## Arguments

- e1:

  First operand (mlx or numeric)

- e2:

  Second operand (mlx or numeric)

## Value

An mlx object

## See also

[mlx.core.array](https://ml-explore.github.io/mlx/build/html/python/array.html)

## Examples

``` r
if (FALSE) { # \dontrun{
x <- as_mlx(matrix(1:4, 2, 2))
y <- as_mlx(matrix(5:8, 2, 2))
x + y
x < y
} # }
```
