# Column means for mlx arrays

Column means for mlx arrays

## Usage

``` r
colMeans(x, ...)

# Default S3 method
colMeans(x, na.rm = FALSE, dims = 1, ...)

# S3 method for class 'mlx'
colMeans(x, na.rm = FALSE, dims = 1, ...)
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

An mlx array if `x` is_mlx, otherwise a numeric vector.

## See also

[mlx.core.mean](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.mean)

## Examples

``` r
x <- mlx_matrix(1:6, 3, 2)
colMeans(x)
#> mlx array [2]
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 2 5
```
