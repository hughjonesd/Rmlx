# Gather elements from an mlx array

Wraps
[`mlx.core.gather()`](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.gather)
so you can pull elements by axis. Provide one index per axis. Axes must
be positive integers (we don't allow negative indices, unlike Python).

## Usage

``` r
mlx_gather(x, indices, axes = NULL)
```

## Arguments

- x:

  An mlx array.

- indices:

  List of numeric/logical vectors or arrays (R or `mlx`). All entries
  must broadcast to a common shape.

- axes:

  Integer vector of axes (1-indexed). Defaults to the first
  `length(indices)` axes.

## Value

An `mlx` array containing the gathered elements.

## Element-wise indexing

The output has the same shape as the indices. Each element of the output
is `x[index_1, index_2, ...]` from the corresponding position of each
index. See the examples below.

## Examples

``` r
x <- mlx_matrix(1:9, 3, 3)

# Simple cartesian gather:
mlx_gather(x, list(1:2, 1:2), axes = 1:2)
#> mlx array [2]
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 1 5

# Element-wise pairs: grab a custom 2x2 grid of coordinates
row_idx <- matrix(c(1, 1,
                    2, 3), nrow = 2, byrow = TRUE)
col_idx <- matrix(c(1, 3,
                    2, 2), nrow = 2, byrow = TRUE)
mlx_gather(x, list(row_idx, col_idx), axes = c(1L, 2L))
#> mlx array [2 x 2]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1] [,2]
#> [1,]    1    7
#> [2,]    5    6
```
