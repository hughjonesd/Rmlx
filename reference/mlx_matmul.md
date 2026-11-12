# Matrix multiplication with broadcast batch dimensions

Unlike `%*%`, `mlx_matmul()` accepts arrays with more than two
dimensions. The last two axes are matrix axes; preceding axes broadcast.
One-dimensional operands are promoted to row/column matrices and the
added axes are removed from the result, following MLX semantics. Scalars
are not supported.

## Usage

``` r
mlx_matmul(x, y)
```

## Arguments

- x:

  An mlx array, or an R array/matrix/vector that will be converted via
  [`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md).

- y:

  A second array, with the same accepted types as `x`.

## Value

An mlx array. Two vectors produce a scalar inner product.

## See also

[mlx.core.matmul](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.matmul.html)

## Examples

``` r
x <- mlx_ones(c(2, 3, 4))
w <- mlx_ones(c(4, 5))
mlx_shape(mlx_matmul(x, w)) # 2 3 5
#> [1] 2 3 5
```
