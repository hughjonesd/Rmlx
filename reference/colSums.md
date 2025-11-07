# Column sums for mlx arrays

Column sums for mlx arrays

## Usage

``` r
colSums(x, ...)

# Default S3 method
colSums(x, na.rm = FALSE, dims = 1, ...)

# S3 method for class 'mlx'
colSums(x, na.rm = FALSE, dims = 1, ...)
```

## Arguments

- x:

  An array or mlx array.

- ...:

  Additional arguments forwarded to the base implementation.

- na.rm:

  Logical; currently ignored for mlx arrays.

- dims:

  Dimensions passed through to the base implementation when `x` is not
  an mlx array.

## Value

An mlx array if `x` is mlx, otherwise a numeric vector.

## See also

[mlx.core.sum](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.sum)

## Examples

``` r
x <- as_mlx(matrix(1:6, 3, 2))
colSums(x)
#> mlx array [2]
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1]  6 15
```
