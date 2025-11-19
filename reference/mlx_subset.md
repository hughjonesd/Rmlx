# Subset MLX array

MLX subsetting mirrors base R for the common cases while avoiding a few
of the language's historical footguns:

## Usage

``` r
# S3 method for class 'mlx'
x[..., drop = FALSE]

# S3 method for class 'mlx'
x[...] <- value
```

## Arguments

- x:

  An mlx array, or an R array/matrix/vector that will be converted via
  [`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md).

- ...:

  Indices for each dimension. Provide one per axis; omitted indices
  select the full extent. Logical indices recycle to the dimension
  length.

- drop:

  Should dimensions be dropped? (default: FALSE)

- value:

  Replacement values, recycled to match the selection.

## Value

The subsetted MLX object.

## Details

- **Numeric indices**: positive (1-based) and purely negative vectors
  are supported. Negative indices drop the listed elements, just as in
  base R. Mixing signs is an error and `0` is not allowed.

- **Logical indices**: recycled to the target dimension length. Logical
  masks may be mixed with numeric indices across dimensions.

- **Matrices/arrays**: numeric matrices (or higher dimensional arrays)
  select individual elements, one coordinate per row. The trailing
  dimension must match the array rank and entries must be positive;
  negative matrices are rejected to avoid ambiguous complements.

- **`mlx` indices**: `mlx` vectors, logical masks, and matrices behave
  the same as their R equivalents. One-dimensional MLX arrays are
  treated as vectors rather than 1-column matrices.

- **`drop`**: dimensions are preserved by default (`drop = FALSE`),
  matching the package's preference for explicit shapes.

- **Unsupported**: character indices and named lookups are not
  implemented.

## See also

[mlx.core.take](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.take)

## Examples

``` r
x <- as_mlx(matrix(1:9, 3, 3))
x[1, ]
#> mlx array [1 x 3]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1] [,2] [,3]
#> [1,]    1    4    7
```
