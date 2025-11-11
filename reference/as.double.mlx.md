# Convert MLX array to numeric vector

Converts an MLX array to a numeric (double) vector, dropping dimensions
and coercing types as needed. Integers and booleans are converted to
doubles.

## Usage

``` r
# S3 method for class 'mlx'
as.double(x, ...)

# S3 method for class 'mlx'
as.numeric(x, ...)
```

## Arguments

- x:

  An mlx array.

- ...:

  Additional arguments (currently unused).

## Value

A numeric vector.

## Examples

``` r
x_int <- as_mlx(c(1L, 2L, 3L), dtype = "int32")
as.numeric(x_int)  # Returns c(1.0, 2.0, 3.0)
#> [1] 1 2 3

x_bool <- as_mlx(c(TRUE, FALSE, TRUE))
as.numeric(x_bool)  # Returns c(1.0, 0.0, 1.0)
#> [1] 1 0 1
```
