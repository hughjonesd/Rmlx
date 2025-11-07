# Gather elements from an mlx array

Mirrors
[`mlx.core.gather()`](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.gather),
selecting elements according to index tensors along the specified axes.
The current implementation supports gathering along a single axis at a
time.

## Usage

``` r
mlx_gather(x, indices, axes = NULL)
```

## Arguments

- x:

  An mlx array.

- indices:

  List of index tensors. Each element can be a numeric/logical vector,
  array, or an `mlx` array of integer type. Shapes must broadcast to a
  common result.

- axes:

  Integer vector of axes (1-indexed, negatives count from the end)
  corresponding to `indices`. Defaults to the first `length(indices)`
  axes.

## Value

An `mlx` array containing the gathered elements.

## Examples

``` r
x <- as_mlx(matrix(1:9, 3, 3))
idx_rows <- c(1L, 3L)
gathered <- mlx_gather(x, list(idx_rows), axes = 1L)
as.matrix(gathered)
#>      [,1] [,2] [,3]
#> [1,]    1    4    7
#> [2,]    3    6    9
```
