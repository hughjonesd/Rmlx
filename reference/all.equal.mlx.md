# Test if two MLX arrays are (nearly) equal

S3 method for `all.equal` following R semantics. Returns `TRUE` if
arrays are close, or a character vector describing differences if they
are not.

## Usage

``` r
# S3 method for class 'mlx'
all.equal(target, current, tolerance = sqrt(.Machine$double.eps), ...)
```

## Arguments

- target, current:

  MLX arrays to compare

- tolerance:

  Numeric tolerance for comparison (default: sqrt(.Machine\$double.eps))

- ...:

  Additional arguments (currently ignored)

## Value

Either `TRUE` or a character vector describing differences

## Details

This method follows R's
[`all.equal()`](https://rdrr.io/r/base/all.equal.html) semantics:

- Returns `TRUE` if arrays are close within tolerance

- Returns a character vector describing differences otherwise

- Checks dimensions/shapes before comparing values

The tolerance is converted to MLX's rtol and atol parameters:

- rtol = tolerance

- atol = tolerance

## See also

[`mlx_allclose()`](https://hughjonesd.github.io/Rmlx/reference/mlx_allclose.md),
[`mlx_isclose()`](https://hughjonesd.github.io/Rmlx/reference/mlx_isclose.md)

## Examples

``` r
a <- as_mlx(c(1.0, 2.0, 3.0))
b <- as_mlx(c(1.0 + 1e-6, 2.0 + 1e-6, 3.0 + 1e-6))
all.equal(a, b)  # TRUE
#> [1] "Arrays are not all close within tolerance"

c <- as_mlx(c(1.0, 2.0, 10.0))
all.equal(a, c)  # Character vector describing difference
#> [1] "Arrays are not all close within tolerance"
```
