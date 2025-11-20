# Object structure for MLX array

Object structure for MLX array

## Usage

``` r
# S3 method for class 'mlx'
str(object, ...)
```

## Arguments

- object:

  An mlx object

- ...:

  Additional arguments (ignored)

## Value

`NULL` invisibly.

## Examples

``` r
x <- mlx_matrix(1:4, 2, 2)
str(x)
#> mlx [2 x 2] float32 on gpu
```
