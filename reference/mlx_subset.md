# Subset MLX array

MLX subsetting is like base R with a few differences:

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

- `drop = FALSE` by default.

- Indices containing `NA` give an error.

- Single indices on a 2D or higher array are only allowed for
  assignment. For example, if `x` is a matrix, `x[x < 0] <- 0` is OK but
  `subset <- x[x < 0]` is not. Use
  [`mlx_flatten()`](https://hughjonesd.github.io/Rmlx/reference/mlx_flatten.md)
  explicitly for subsetting.

- There is one exception: as in R, a single numeric matrix index selects
  individual elements. The number of columns must match the rank of `x`;
  each row gives coordinates for one element. The return value from
  subsetting is a flat mlx vector.

- `mlx` vectors, logical masks, and matrices behave the same as their R
  equivalents.

- Duplicate assignments like `x[c(1,1)] <- 2:3` are undefined behaviour.

- Character indices are not supported as MLX has no concept of dimension
  names.

## See also

[mlx.core.take](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.take)

## Examples

``` r
x <- mlx_matrix(1:9, 3, 3)
x[1, ]
#> mlx array [1 x 3]
#>   dtype: float32
#>   values:
#>      [,1] [,2] [,3]
#> [1,]    1    4    7
```
