# Convert MLX array to R vector

Converts an MLX array to an R vector. Multi-dimensional arrays are
flattened in column-major order (R's default).

## Usage

``` r
# S3 method for class 'mlx'
as.vector(x, mode = "any")

# S3 method for class 'mlx'
as.logical(x, ...)

# S3 method for class 'mlx'
as.double(x, ...)

# S3 method for class 'mlx'
as.numeric(x, ...)

# S3 method for class 'mlx'
as.integer(x, ...)
```

## Arguments

- x:

  An mlx array.

- mode:

  Character string specifying the type of vector to return (passed to
  [`base::as.vector()`](https://rdrr.io/r/base/vector.html))

- ...:

  Additional arguments (ignored)

## Value

A vector of the specified mode.

## Examples

``` r
x <- as_mlx(-1:1)
as.vector(x)
#> [1] -1  0  1
as.logical(x)
#> [1]  TRUE FALSE  TRUE
as.numeric(x)
#> [1] -1  0  1

# Multi-dimensional arrays are flattened
m <- mlx_matrix(1:6, 2, 3)
as.vector(m)  # Flattened in column-major order
#> [1] 1 2 3 4 5 6
```
