# Remove singleton dimensions

Remove singleton dimensions

## Usage

``` r
mlx_squeeze(x, axis = NULL)
```

## Arguments

- x:

  An mlx array.

- axis:

  Optional integer vector of axes (1-indexed) to remove. When `NULL` all
  axes of length one are removed.

## Value

An mlx array with the selected axes removed.

## See also

[mlx.core.squeeze](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.squeeze)

## Examples

``` r
x <- as_mlx(array(1:4, dim = c(1, 2, 2, 1)))
mlx_squeeze(x)
#> mlx array [2 x 2]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1] [,2]
#> [1,]    1    3
#> [2,]    2    4
mlx_squeeze(x, axis = 1)
#> mlx array [2 x 2 x 1]
#>   dtype: float32
#>   device: gpu
#>   (4 elements, not shown)
```
