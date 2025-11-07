# Convert between radians and degrees

`mlx_degrees()` and `mlx_radians()` mirror
[`mlx.core.degrees()`](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.degrees)
and
[`mlx.core.radians()`](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.radians),
converting angular values elementwise using MLX kernels.

## Usage

``` r
mlx_degrees(x)

mlx_radians(x)
```

## Arguments

- x:

  An mlx array.

## Value

An mlx array with transformed angular units.

## See also

[mlx.core.degrees](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.degrees),
[mlx.core.radians](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.radians)

## Examples

``` r
x <- as_mlx(pi / 2)
as.matrix(mlx_degrees(x))  # 90
#> [1] 90
angles <- mlx_radians(as_mlx(c(0, 90, 180)))
as.matrix(angles)
#> [1] 0.000000 1.570796 3.141593
```
