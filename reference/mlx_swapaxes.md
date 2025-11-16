# Swap two axes of an mlx array

`mlx_swapaxes()` mirrors
[`mlx.core.swapaxes()`](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.swapaxes),
exchanging two dimensions while leaving others intact.

## Usage

``` r
mlx_swapaxes(x, axis1, axis2)
```

## Arguments

- x:

  An mlx array.

- axis1, axis2:

  Axes to swap (1-indexed).

## Value

An mlx array with the specified axes exchanged.

## See also

[mlx.core.swapaxes](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.swapaxes)

## Examples

``` r
x <- as_mlx(array(1:24, dim = c(2, 3, 4)))
swapped <- mlx_swapaxes(x, axis1 = 1, axis2 = 3)
dim(swapped)
#> [1] 4 3 2
```
