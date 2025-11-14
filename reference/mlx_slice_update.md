# Update a slice of an mlx array

Wrapper around
[`mlx.core.slice_update()`](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.slice_update)
that replaces a contiguous strided region with `value`.

## Usage

``` r
mlx_slice_update(x, value, start, stop, strides = NULL)
```

## Arguments

- x:

  An mlx array.

- value:

  Replacement `mlx` (or coercible) array. Must broadcast to the slice
  determined by `start`, `stop`, and `strides`.

- start:

  Integer vector (1-indexed) giving the inclusive starting index for
  each axis.

- stop:

  Integer vector (1-indexed) giving the inclusive stopping index for
  each axis.

- strides:

  Optional integer vector of strides (defaults to ones).

## Value

An `mlx` array with the specified slice replaced.

## Examples

``` r
x <- as_mlx(matrix(1:9, 3, 3))
replacement <- as_mlx(matrix(100:103, nrow = 2))
updated <- mlx_slice_update(x, replacement, start = c(1L, 2L), stop = c(2L, 3L))
as.matrix(updated)
#>      [,1] [,2] [,3]
#> [1,]    1  100  102
#> [2,]    2  101  103
#> [3,]    3    6    9
```
