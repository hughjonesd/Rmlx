# Unflatten an axis into multiple axes

The reverse of flattening: expands a single axis into multiple axes with
the given shape.

## Usage

``` r
mlx_unflatten(x, axis, shape)
```

## Arguments

- x:

  An mlx array.

- axis:

  Which axis to unflatten (1-indexed).

- shape:

  Integer vector specifying the new shape for the unflattened axis.

## Value

An mlx array with the axis expanded.

## See also

[mlx.core.unflatten](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.unflatten.html)

## Examples

``` r
# Flatten and unflatten
x <- as_mlx(array(1:24, c(2, 3, 4)))
x_flat <- mlx_reshape(x, c(2, 12))  # flatten last two dims
mlx_unflatten(x_flat, axis = 2, shape = c(3, 4))  # restore original shape
#> mlx array [2 x 3 x 4]
#>   dtype: float32
#>   device: gpu
#>   (24 elements, not shown)
```
