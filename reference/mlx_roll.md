# Roll array elements

Roll array elements

## Usage

``` r
mlx_roll(x, shift, axis = NULL)
```

## Arguments

- x:

  An mlx array.

- shift:

  Integer vector giving the number of places by which elements are
  shifted.

- axis:

  Optional axis (or axes) along which elements are shifted. When `NULL`,
  the array is flattened and shifted.

## Value

An mlx array with elements circularly shifted.

## See also

[mlx.core.roll](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.roll)

## Examples

``` r
x <- as_mlx(matrix(1:4, 2, 2))
mlx_roll(x, shift = 1, axis = 2)
#> mlx array [2 x 2]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1] [,2]
#> [1,]    3    1
#> [2,]    4    2
```
