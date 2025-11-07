# Hadamard transform for MLX arrays

Multiplies the last dimension of `x` by the Sylvester-Hadamard matrix of
the corresponding size. The transform expects the length of the last
axis to be a power of two.

## Usage

``` r
mlx_hadamard_transform(x, scale = NULL)
```

## Arguments

- x:

  An mlx array, or an R array/matrix/vector that will be converted via
  [`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md).

- scale:

  Optional numeric scalar applied to the result. MLX defaults to
  `1 / sqrt(n)` where `n` is the size of the transformed axis; set
  `scale` to override the factor (for example, `scale = 1` yields the
  unnormalised Hadamard transform).

## Value

An `mlx` array containing the Hadamard-transformed values.

## See also

<https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.hadamard_transform>

## Examples

``` r
x <- as_mlx(c(1, -1))
as.vector(mlx_hadamard_transform(x))
#> [1] 0.000000 1.414214
as.vector(mlx_hadamard_transform(x, scale = 1))
#> [1] 0 2
```
