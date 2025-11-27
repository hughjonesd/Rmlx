# Row sums for mlx arrays

Row sums for mlx arrays

## Usage

``` r
rowSums(x, ...)

# Default S3 method
rowSums(x, na.rm = FALSE, dims = 1, ...)

# S3 method for class 'mlx'
rowSums(x, na.rm = FALSE, dims = 1, ...)
```

## Arguments

- x:

  An array or mlx array.

- ...:

  Additional arguments forwarded to the corresponding base R
  implementation for signature compatibility.

- na.rm:

  Logical; currently ignored for mlx arrays.

- dims:

  Leading dimensions treated as rows/cols (see
  [`base::rowSums()`](https://rdrr.io/r/base/colSums.html)).

## Value

An mlx array if `x` is_mlx, otherwise a numeric vector.

## See also

[mlx.core.sum](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.sum)

## Examples

``` r
x <- mlx_matrix(1:6, 3, 2)
rowSums(x)
#> mlx array [3]
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 5 7 9
```
