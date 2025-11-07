# Set dimensions of MLX array

Reshapes the MLX array to the specified dimensions. The total number of
elements must remain the same.

## Usage

``` r
# S3 method for class 'mlx'
dim(x) <- value
```

## Arguments

- x:

  An mlx array, or an R array/matrix/vector that will be converted via
  [`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md).

- value:

  Integer vector of new dimensions

## Value

Reshaped mlx object

## Examples

``` r
x <- as_mlx(1:12)
dim(x) <- c(3, 4)
dim(x)
#> [1] 3 4
```
