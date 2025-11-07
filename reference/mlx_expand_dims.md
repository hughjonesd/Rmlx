# Insert singleton dimensions

Insert singleton dimensions

## Usage

``` r
mlx_expand_dims(x, axis)
```

## Arguments

- x:

  An mlx array.

- axis:

  Integer vector of axis positions (1-indexed) where new singleton
  dimensions should be inserted.

## Value

An mlx array with additional dimensions of length one.

## See also

[mlx.core.expand_dims](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.expand_dims)

## Examples

``` r
x <- as_mlx(matrix(1:4, 2, 2))
mlx_expand_dims(x, axis = 1)
#> mlx array [1 x 2 x 2]
#>   dtype: float32
#>   device: gpu
#>   (4 elements, not shown)
```
