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

## Difference from Python/C++

Unlike Python's slice notation `array[start:stop]` which uses an
exclusive upper bound, `mlx_slice_update()` uses **inclusive** bounds
for both `start` and `stop` to match R conventions and to be consistent
with
[`mlx_arange()`](https://hughjonesd.github.io/Rmlx/reference/mlx_arange.md)
and
[`mlx_linspace()`](https://hughjonesd.github.io/Rmlx/reference/mlx_linspace.md).

## Examples

``` r
x <- mlx_matrix(1:9, 3, 3)
replacement <- mlx_matrix(100:103, nrow = 2)
updated <- mlx_slice_update(x, replacement, start = c(1L, 2L), stop = c(2L, 3L))
updated
#> mlx array [3 x 3]
#>   dtype: float32
#>   values:
#>      [,1] [,2] [,3]
#> [1,]    1  100  102
#> [2,]    2  101  103
#> [3,]    3    6    9
```
