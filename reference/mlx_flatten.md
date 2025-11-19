# Flatten axes of an mlx array

`mlx_flatten()` mirrors
[`mlx.core.flatten()`](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.flatten),
collapsing a contiguous range of axes into a single dimension.

## Usage

``` r
mlx_flatten(x, start_axis = 1L, end_axis = NULL)
```

## Arguments

- x:

  An mlx array.

- start_axis:

  First axis (1-indexed) in the flattened range.

- end_axis:

  Last axis (1-indexed) in the flattened range. Omit to use the final
  dimension.

## Value

An mlx array with the selected axes collapsed.

## See also

[mlx.core.flatten](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.flatten)

## Examples

``` r
x <- as_mlx(array(1:12, dim = c(2, 3, 2)))
mlx_flatten(x)
#> mlx array [12]
#>   dtype: float32
#>   device: gpu
#>   values:
#>  [1]  1  7  3  9  5 11  2  8  4 10  6 12
mlx_flatten(x, start_axis = 2, end_axis = 3)
#> mlx array [2 x 6]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1] [,2] [,3] [,4] [,5] [,6]
#> [1,]    1    7    3    9    5   11
#> [2,]    2    8    4   10    6   12
```
