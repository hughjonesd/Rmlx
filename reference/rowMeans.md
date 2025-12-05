# Row means for mlx arrays

Row means for mlx arrays

## Usage

``` r
rowMeans(x, ...)

# Default S3 method
rowMeans(x, na.rm = FALSE, dims = 1, ...)

# S3 method for class 'mlx'
rowMeans(x, na.rm = FALSE, dims = 1, ...)
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

[mlx.core.mean](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.mean)

## Examples

``` r
x <- mlx_matrix(1:6, 3, 2)
rowMeans(x)
#> mlx array [3]
#>   dtype: float32
#>   device: cpu
#>   values:
#> [1] 2.5 3.5 4.5
```
