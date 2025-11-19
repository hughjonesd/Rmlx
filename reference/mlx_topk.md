# Top-k selection and partitioning on mlx arrays

`mlx_topk()` returns the largest `k` values. `mlx_partition()` and
`mlx_argpartition()` perform partial sorting, rearranging elements so
that the element at position `kth` is in its correctly sorted position,
with all smaller elements before it and all larger elements after it.
This is more efficient than full sorting when you only need elements up
to a certain rank.

## Usage

``` r
mlx_topk(x, k, axis = NULL)

mlx_partition(x, kth, axis = NULL)

mlx_argpartition(x, kth, axis = NULL)
```

## Arguments

- x:

  An mlx array, or an R array/matrix/vector that will be converted via
  [`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md).

- k:

  Positive integer specifying the number of elements to select.

- axis:

  Single axis (1-indexed). Supply a positive integer between 1 and the
  array rank. Use `NULL` when the helper interprets it as "all axes"
  (see individual docs).

- kth:

  Zero-based index of the element that should be placed in-order after
  partitioning.

## Value

An mlx array. For `mlx_argpartition()`, returns 1-based indices
(following R conventions) showing the partition ordering.

## Details

- `mlx_topk()` returns the largest `k` values along the specified axis.

- `mlx_partition()` rearranges elements so the kth element is correctly
  positioned.

- `mlx_argpartition()` returns the **1-based indices** that would
  partition the array. This follows R's indexing convention (unlike the
  underlying MLX library which uses 0-based indexing).

Use
[`mlx_argsort()`](https://hughjonesd.github.io/Rmlx/reference/mlx_sort.md)
if you need fully sorted indices.

## See also

[mlx.core.topk](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.topk),
[mlx.core.partition](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.partition),
[mlx.core.argpartition](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.argpartition)

## Examples

``` r
scores <- as_mlx(c(0.7, 0.2, 0.9, 0.4))
mlx_topk(scores, k = 2)
#> mlx array [2]
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 0.7 0.9
mlx_partition(scores, kth = 1)
#> mlx array [4]
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 0.2 0.4 0.7 0.9

# Returns 1-based indices
idx <- mlx_argpartition(scores, kth = 1)
as.integer(as.matrix(idx))  # 1-based indices
#> Warning: Converting array to 1-column matrix
#> [1] 2 4 1 3

mlx_topk(as_mlx(matrix(1:6, 2, 3)), k = 1, axis = 1)
#> mlx array [1 x 3]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1] [,2] [,3]
#> [1,]    2    4    6
```
