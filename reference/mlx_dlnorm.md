# Lognormal distribution functions

Compute density (`mlx_dlnorm`), cumulative distribution (`mlx_plnorm`),
and quantile (`mlx_qlnorm`) functions for the lognormal distribution
using MLX.

## Usage

``` r
mlx_dlnorm(
  x,
  meanlog = 0,
  sdlog = 1,
  log = FALSE,
  device = mlx_default_device()
)

mlx_plnorm(x, meanlog = 0, sdlog = 1, device = mlx_default_device())

mlx_qlnorm(p, meanlog = 0, sdlog = 1, device = mlx_default_device())
```

## Arguments

- x:

  Vector of quantiles (mlx array or coercible to mlx)

- meanlog, sdlog:

  Mean and standard deviation of distribution on log scale (default: 0,
  1)

- log:

  If `TRUE`, return log density for `mlx_dlnorm` (default: `FALSE`)

- device:

  Execution target: supply `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  By default, many functions use the
  [`mlx_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_device.md)
  of their first argument.

- p:

  Vector of probabilities (mlx array or coercible to mlx)

## Value

An mlx array with the computed values.

## Examples

``` r
x <- as_mlx(seq(0.1, 3, by = 0.2))
mlx_dlnorm(x)
#> mlx array [15]
#>   dtype: float32
#>   device: cpu
#>   values:
#>  [1] 0.28159019 0.64420325 0.62749612 0.53479487 0.44081569 0.36103126
#>  [7] 0.29649639 0.24497364 0.20385426 0.17088225 0.14426385 0.12261371
#> [13] 0.10487106 0.09022354 0.07804624
mlx_plnorm(x)
#> mlx array [15]
#>   dtype: float32
#>   device: cpu
#>   values:
#>  [1] 0.01065108 0.11430013 0.24410856 0.36066771 0.45804483 0.53796583
#>  [7] 0.60347962 0.65743208 0.70216185 0.73951596 0.77093738 0.79755211
#> [13] 0.82024282 0.83970636 0.85649657

p <- as_mlx(c(0.25, 0.5, 0.75))
mlx_qlnorm(p)
#> mlx array [3]
#>   dtype: float32
#>   device: cpu
#>   values:
#> [1] 0.5094163 1.0000000 1.9630311
```
