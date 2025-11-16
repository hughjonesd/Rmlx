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
  Defaults to the current
  [`mlx_default_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_default_device.md)
  unless noted otherwise (helpers that act on an existing array
  typically reuse that array's device or stream).

- p:

  Vector of probabilities (mlx array or coercible to mlx)

## Value

An mlx array with the computed values

## Examples

``` r
x <- as_mlx(seq(0.1, 3, by = 0.2))
as.matrix(mlx_dlnorm(x))
#>  [1] 0.28159016 0.64420331 0.62749606 0.53479487 0.44081569 0.36103126
#>  [7] 0.29649639 0.24497366 0.20385425 0.17088225 0.14426388 0.12261371
#> [13] 0.10487107 0.09022354 0.07804624
as.matrix(mlx_plnorm(x))
#>  [1] 0.01065108 0.11430004 0.24410859 0.36066759 0.45804486 0.53796577
#>  [7] 0.60347968 0.65743220 0.70216179 0.73951596 0.77093732 0.79755199
#> [13] 0.82024282 0.83970642 0.85649657

p <- as_mlx(c(0.25, 0.5, 0.75))
as.matrix(mlx_qlnorm(p))
#> [1] 0.5094163 1.0000000 1.9630311
```
