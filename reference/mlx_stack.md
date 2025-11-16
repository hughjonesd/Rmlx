# Stack mlx arrays along a new axis

Stack mlx arrays along a new axis

## Usage

``` r
mlx_stack(..., axis = 1L)
```

## Arguments

- ...:

  One or more arrays (or a single list of arrays) coercible to mlx.

- axis:

  Position of the new axis (1-indexed). Supply values between 1 and
  `length(dim(x)) + 1` to insert anywhere along the dimension list.

## Value

An mlx array with one additional dimension.

## See also

[mlx.core.stack](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.stack)

## Examples

``` r
x <- as_mlx(matrix(1:4, 2, 2))
y <- as_mlx(matrix(5:8, 2, 2))
stacked <- mlx_stack(x, y, axis = 1)
```
