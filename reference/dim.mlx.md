# Get dimensions of MLX array

[`dim()`](https://rdrr.io/r/base/dim.html) mirrors base R semantics and
returns `NULL` for 1-D vectors and scalars, while `mlx_shape()` always
returns the raw MLX shape (integers, never `NULL`). Use `mlx_shape()`
when you need the underlying MLX dimension metadata and
[`dim()`](https://rdrr.io/r/base/dim.html) when you want R-like
behaviour.

## Usage

``` r
# S3 method for class 'mlx'
dim(x)

mlx_shape(x)
```

## Arguments

- x:

  An mlx array, or an R array/matrix/vector that will be converted via
  [`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md).

## Value

For [`dim()`](https://rdrr.io/r/base/dim.html), an integer vector of
dimensions or `NULL` for vectors/ scalars. For `mlx_shape()`, an integer
vector (length zero for scalars).

## Examples

``` r
x <- as_mlx(matrix(1:4, 2, 2))
dim(x)
#> [1] 2 2

v <- as_mlx(1:3)
dim(v)        # NULL (matches base R)
#> NULL
mlx_shape(v)  # 3
#> [1] 3
```
