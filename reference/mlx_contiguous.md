# Ensure contiguous memory layout

Returns a copy of `x` with contiguous strides.

## Usage

``` r
mlx_contiguous(x)
```

## Arguments

- x:

  An mlx array.

## Value

An mlx array backed by contiguous storage.

## See also

<https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.contiguous>

## Examples

``` r
x <- mlx_swapaxes(mlx_matrix(1:4, 2, 2), axis1 = 1, axis2 = 2)
y <- mlx_contiguous(x)
identical(as.array(x), as.array(y))
#> [1] TRUE
```
