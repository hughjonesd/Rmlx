# Extract diagonal or construct diagonal matrix for mlx arrays

Extract a diagonal from a matrix or construct a diagonal matrix from a
vector.

## Usage

``` r
# S3 method for class 'mlx'
diag(x, nrow, ncol, names = TRUE)

mlx_diagonal(x, offset = 0L, axis1 = 1L, axis2 = 2L)
```

## Arguments

- x:

  An mlx array. If 1D, creates a diagonal matrix. If 2D or higher,
  extracts the diagonal.

- nrow, ncol:

  Diagonal offset (nrow only; ncol ignored).

  `diag.mlx()` is an R interface to `mlx_diagonal()` with the same
  semantics as [`base::diag()`](https://rdrr.io/r/base/diag.html).

- names:

  Unused.

- offset:

  Diagonal offset (0 for main diagonal, positive for above, negative for
  below).

- axis1, axis2:

  For multi-dimensional arrays, which axes define the 2D planes
  (1-indexed).

## Value

An mlx array.

## See also

[mlx.core.diagonal](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.diagonal.html)

## Examples

``` r
# Extract diagonal
x <- mlx_matrix(1:9, 3, 3)
mlx_diagonal(x)
#> mlx array [3]
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 1 5 9
# (Constructing diagonals from 1D inputs is not yet supported.)
```
