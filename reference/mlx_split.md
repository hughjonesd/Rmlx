# Split mlx arrays

`mlx_split()` divides an array along an axis either into equal sections
(`sections` scalar) or at explicit 1-based split points (`sections`
vector), returning a list of mlx arrays.

## Usage

``` r
mlx_split(x, sections, axis = 1L)
```

## Arguments

- x:

  An mlx array, or an R array/matrix/vector that will be converted via
  [`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md).

- sections:

  Either a single integer (number of equal parts) or an integer vector
  of 1-based split points along `axis`.

- axis:

  Axis (1-indexed) to operate on.

## Value

A list of mlx arrays split along the chosen axis.

## See also

[mlx.core.split](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.split),
[`mlx_pad()`](https://hughjonesd.github.io/Rmlx/reference/mlx_pad.md)

## Examples

``` r
x <- mlx_matrix(1:4, 2, 2)
parts <- mlx_split(x, sections = 2, axis = 1)
custom_parts <- mlx_split(x, sections = c(1), axis = 2)
```
