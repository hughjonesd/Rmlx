# Generate random permutations on mlx arrays

Generate a random permutation of integers or permute the entries of an
array along a specified axis.

## Usage

``` r
mlx_rand_permutation(x, axis = 1L)
```

## Arguments

- x:

  Either an integer n (to generate a permutation of 1:n), or an mlx
  array or matrix to permute.

- axis:

  Axis (1-indexed) along which to permute when `x` is an array. Default
  is 1L (permute rows).

## Value

An mlx array containing the random permutation.

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
