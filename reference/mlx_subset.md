# Subset MLX array

MLX subsetting mirrors base R for the common cases while avoiding a few
of the language's historical footguns:

## Usage

``` r
# S3 method for class 'mlx'
x[...] <- value

# S3 method for class 'mlx'
x[..., drop = FALSE]
```

## Arguments

- x:

  An mlx array, or an R array/matrix/vector that will be converted via
  [`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md).

- ...:

  Indices for each dimension. Provide one per axis; omitted indices
  select the full extent. Logical indices recycle to the dimension
  length.

- value:

  Value to assign, typically an mlx or R array

- drop:

  Should dimensions be dropped? (default: FALSE)

## Value

The subsetted MLX object.

## Details

- **`drop`**: dimensions are preserved by default (`drop = FALSE`).

- **Numeric indices**: positive (1-based) and purely negative vectors
  are supported. Negative indices drop the listed elements, just as in
  base R. Mixing signs is an error and `0` is not allowed.

- **Logical indices**: recycled to the target dimension length. Logical
  indices may be mixed with numeric indices across dimensions.

- **Flattening indices**: single indices on a 2D or higher array are
  only allowed for assignment. For example, if `x` is a matrix,
  `x[x < 0] <- 0` is fine but `subset <- x[x < 0]` is not. Use
  [`mlx_flatten()`](https://hughjonesd.github.io/Rmlx/reference/mlx_flatten.md)
  explicitly for subsetting.

- **NA values**: indices containing `NA` are rejected with an error.

- **Matrix indices**: a single numeric matrix index selects individual
  elements. The number of columns must match the rank of `x`; each row
  gives coordinates for one element.

- **`mlx` indices**: `mlx` vectors, logical masks, and matrices behave
  the same as their R equivalents. One-dimensional MLX arrays are
  treated as vectors rather than 1-column matrices.

- **Duplicates**: duplicate assignments like `x[c(1,1)] <- 2:3` give an
  error.

- **Unsupported**: character indices and named lookups are not
  implemented.

## See also

[mlx.core.take](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.take)

## Examples

``` r
x <- mlx_matrix(1:9, 3, 3)
x[1, ]
#> mlx array [1 x 3]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1] [,2] [,3]
#> [1,]    1    4    7
```
