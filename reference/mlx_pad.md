# Pad or split mlx arrays

- `mlx_pad()` mirrors the MLX padding primitive, enlarging each axis
  according to `pad_width`. Values are added symmetrically
  (`pad_width[i, 1]` before, `pad_width[i, 2]` after) using the
  specified `mode`.

- `mlx_split()` divides an array along an axis either into equal
  sections (`sections` scalar) or at explicit 1-based split points
  (`sections` vector), returning a list of mlx arrays.

## Usage

``` r
mlx_pad(
  x,
  pad_width,
  value = 0,
  mode = c("constant", "edge", "reflect", "symmetric"),
  axes = NULL
)

mlx_split(x, sections, axis = 1L)
```

## Arguments

- x:

  An mlx array, or an R array/matrix/vector that will be converted via
  [`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md).

- pad_width:

  Padding extents. Supply a single integer, a length-two numeric vector,
  or a matrix/list with one `(before, after)` pair per padded axis.

- value:

  Constant fill value used when `mode = "constant"`.

- mode:

  Padding mode passed to MLX (e.g., `"constant"`, `"edge"`,
  `"reflect"`).

- axes:

  Optional integer vector of axes (1-indexed, negatives count from the
  end) to which `pad_width` applies. Unlisted axes receive zero padding.

- sections:

  Either a single integer (number of equal parts) or an integer vector
  of 1-based split points along `axis`.

- axis:

  Axis (1-indexed, negatives count from the end) to operate on.

## Value

For `mlx_pad()`, an mlx array; for `mlx_split()`, a list of mlx arrays.

## See also

[mlx.core.pad](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.pad)

[mlx.core.split](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.split)

## Examples

``` r
x <- as_mlx(matrix(1:4, 2, 2))
padded <- mlx_pad(x, pad_width = 1)
padded_cols <- mlx_pad(x, pad_width = c(0, 1), axes = 2)
parts <- mlx_split(x, sections = 2, axis = 1)
custom_parts <- mlx_split(x, sections = c(1), axis = 2)
```
