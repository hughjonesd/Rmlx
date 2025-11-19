# Automatic differentiation for MLX functions

`mlx_grad()` computes gradients of an R function that operates on mlx
arrays. The function must keep all differentiable computations in MLX
(e.g., via
[`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md) and
MLX operators) and return an mlx object.

## Usage

``` r
mlx_grad(f, ..., argnums = NULL, value = FALSE)

mlx_value_grad(f, ..., argnums = NULL)
```

## Arguments

- f:

  An R function. Its arguments should be mlx objects, and its return
  value must be an mlx array (typically a scalar loss).

- ...:

  Arguments to pass to `f`. They will be coerced to mlx if needed.

- argnums:

  Indices (1-based) identifying which arguments to differentiate with
  respect to. Defaults to all arguments.

- value:

  Should the function value be returned alongside gradients? Set to
  `TRUE` to receive a list with components `value` and `grads`.

## Value

When `value = FALSE` (default), a list of mlx arrays containing the
gradients in the same order as `argnums`. When `value = TRUE`, a list
with elements `value` (the function output as mlx) and `grads`.

## Details

Keep the differentiated closure inside MLX operations. Coercing arrays
back to base R objects (e.g. via
[`as.matrix()`](https://rdrr.io/r/base/matrix.html) or `[[` extraction)
breaks the gradient tape and results in an error.

## See also

[mlx.core.grad](https://ml-explore.github.io/mlx/build/html/python/transforms.html#mlx.core.grad),
[mlx.core.value_and_grad](https://ml-explore.github.io/mlx/build/html/python/transforms.html#mlx.core.value_and_grad)

## Examples

``` r
loss <- function(w, x, y) {
  preds <- x %*% w
  resids <- preds - y
  sum(resids * resids) / length(y)
}
x <- as_mlx(matrix(1:8, 4, 2))
y <- as_mlx(matrix(c(1, 3, 2, 4), 4, 1))
w <- as_mlx(matrix(0, 2, 1))
mlx_grad(loss, w, x, y)[[1]]
#> mlx array [2 x 1]
#>   dtype: float32
#>   device: gpu
#>   values:
#>       [,1]
#> [1,] -14.5
#> [2,] -34.5
loss <- function(w, x) sum((x %*% w) * (x %*% w))
x <- as_mlx(matrix(1:4, 2, 2))
w <- as_mlx(matrix(c(1, -1), 2, 1))
mlx_value_grad(loss, w, x)
#> $value
#> mlx array []
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 8
#> 
#> $grads
#> $grads[[1]]
#> mlx array [2 x 1]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1]
#> [1,]  -12
#> [2,]  -28
#> 
#> $grads[[2]]
#> mlx array [2 x 2]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1] [,2]
#> [1,]   -4    4
#> [2,]   -4    4
#> 
#> 
```
