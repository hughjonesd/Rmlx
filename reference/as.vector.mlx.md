# Convert MLX array to R vector

Converts an MLX array to an R vector. For multi-dimensional arrays (2+
dimensions), a warning is issued and the array is flattened in
column-major order (R's default).

## Usage

``` r
# S3 method for class 'mlx'
as.vector(x, mode = "any")
```

## Arguments

- x:

  An mlx array.

- mode:

  Character string specifying the type of vector to return (passed to
  [`base::as.vector()`](https://rdrr.io/r/base/vector.html))

## Value

A vector of the specified mode

## Examples

``` r
x <- as_mlx(1:5)
as.vector(x)
#> [1] 1 2 3 4 5

# Multi-dimensional arrays produce a warning
m <- as_mlx(matrix(1:6, 2, 3))
v <- suppressWarnings(as.vector(m))  # Flattened in column-major order
```
