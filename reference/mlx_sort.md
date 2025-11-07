# Sort and argsort for mlx arrays

`mlx_sort()` returns sorted values along the specified axis.
`mlx_argsort()` returns the indices that would sort the array.

## Usage

``` r
mlx_sort(x, axis = NULL)

mlx_argsort(x, axis = NULL)
```

## Arguments

- x:

  An mlx array, or an R array/matrix/vector that will be converted via
  [`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md).

- axis:

  Optional axis to operate over (1-indexed like R). When `NULL`, the
  array is flattened first.

## Value

An mlx array containing sorted values (for `mlx_sort()`) or **1-based
indices** (for `mlx_argsort()`). The indices follow R's indexing
convention and can be used directly with R's `[` operator.

## Details

`mlx_argsort()` returns **1-based indices** that would sort the array in
ascending order. This follows R's indexing convention (unlike the
underlying MLX library which uses 0-based indexing). The returned
indices can be used directly to reorder the original array.

For partial sorting (finding elements up to a certain rank without fully
sorting), see
[`mlx_partition()`](https://hughjonesd.github.io/Rmlx/reference/mlx_topk.md)
and
[`mlx_argpartition()`](https://hughjonesd.github.io/Rmlx/reference/mlx_topk.md).

## See also

[mlx.core.sort](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.sort),
[mlx.core.argsort](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.argsort)

## Examples

``` r
x <- as_mlx(c(3, 1, 4, 2))
mlx_sort(x)
#> mlx array [4]
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 1 2 3 4

# Returns 1-based indices
idx <- mlx_argsort(x)
as.integer(as.matrix(idx))  # [1] 2 4 1 3
#> [1] 2 4 1 3

# Can be used directly with R indexing
original <- c(3, 1, 4, 2)
sorted_idx <- as.integer(as.matrix(mlx_argsort(as_mlx(original))))
original[sorted_idx]  # [1] 1 2 3 4
#> [1] 1 2 3 4

mlx_sort(as_mlx(matrix(1:6, 2, 3)), axis = 1)
#> mlx array [2 x 3]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1] [,2] [,3]
#> [1,]    1    3    5
#> [2,]    2    4    6
```
