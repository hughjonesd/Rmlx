# Generate random permutations on mlx arrays

Generate a random permutation of integers or permute the entries of an
array along a specified axis.

## Usage

``` r
mlx_rand_permutation(x, axis = 1L, device = mlx_default_device())
```

## Arguments

- x:

  Either an integer n (to generate a permutation of 1:n), or an mlx
  array or matrix to permute.

- axis:

  Axis (1-indexed) along which to permute when `x` is an array. Default
  is 1L (permute rows).

- device:

  Execution target: supply `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  Defaults to the current
  [`mlx_default_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_default_device.md)
  unless noted otherwise (helpers that act on an existing array
  typically reuse that array's device or stream).

## Value

An mlx array containing the random permutation.

## Details

When `x` is an integer, the result is created on the specified device or
stream; otherwise the permutation follows the input array's device.

## See also

[mlx.core.random.permutation](https://ml-explore.github.io/mlx/build/html/python/random.html#mlx.core.random.permutation)

## Examples

``` r
# Generate a random permutation of 1:10
perm <- mlx_rand_permutation(10)

# Permute the rows of a matrix
mat <- matrix(1:12, 4, 3)
perm_mat <- mlx_rand_permutation(mat)

# Permute columns instead
perm_cols <- mlx_rand_permutation(mat, axis = 2)
```
