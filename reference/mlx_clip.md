# Clip mlx array values into a range

Clip mlx array values into a range

## Usage

``` r
mlx_clip(x, min = NULL, max = NULL)
```

## Arguments

- x:

  An mlx array, or an R array/matrix/vector that will be converted via
  [`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md).

- min, max:

  Scalar bounds. Use `NULL` to leave a bound open.

## Value

An mlx array with values clipped to `[min, max]`.

## See also

[mlx.core.clip](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.clip)

## Examples

``` r
x <- as_mlx(rnorm(4))
mlx_clip(x, min = -1, max = 1)
#> mlx array [4]
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] -0.1557955 -1.0000000 -0.4781501  0.4179416
```
