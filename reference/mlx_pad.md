# Pad mlx arrays

`mlx_pad()` mirrors the MLX padding primitive, enlarging each axis
according to `pad_width`. Values are added symmetrically
(`pad_width[i, 1]` before, `pad_width[i, 2]` after) using the specified
`mode`.

## Usage

``` r
mlx_pad(
  x,
  pad_width,
  value = 0,
  mode = c("constant", "edge", "reflect", "symmetric"),
  axes = NULL
)
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

  Optional integer vector of axes (1-indexed) to which `pad_width`
  applies. Unlisted axes receive zero padding.

## Value

An mlx array with the requested padding applied.

## See also

[mlx.core.pad](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.pad),
[`mlx_split()`](https://hughjonesd.github.io/Rmlx/reference/mlx_split.md)

## Examples

``` r
x <- mlx_matrix(1:4, 2, 2)
padded <- mlx_pad(x, pad_width = 1)
padded_cols <- mlx_pad(x, pad_width = c(0, 1), axes = 2)
```
