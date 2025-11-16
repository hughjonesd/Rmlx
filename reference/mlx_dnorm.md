# Normal distribution functions

Compute density (`mlx_dnorm`), cumulative distribution (`mlx_pnorm`),
and quantile (`mlx_qnorm`) functions for the normal distribution using
MLX.

## Usage

``` r
mlx_dnorm(x, mean = 0, sd = 1, log = FALSE, device = mlx_default_device())

mlx_pnorm(x, mean = 0, sd = 1, device = mlx_default_device())

mlx_qnorm(p, mean = 0, sd = 1, device = mlx_default_device())
```

## Arguments

- x:

  Vector of quantiles (mlx array or coercible to mlx)

- mean:

  Mean of the distribution (default: 0)

- sd:

  Standard deviation of the distribution (default: 1)

- log:

  If `TRUE`, return log density for `mlx_dnorm` (default: `FALSE`)

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

## See also

[`mlx_erf()`](https://hughjonesd.github.io/Rmlx/reference/mlx_erf.md),
[`mlx_erfinv()`](https://hughjonesd.github.io/Rmlx/reference/mlx_erf.md),
[mlx.core.erf](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.erf.html),
[mlx.core.erfinv](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.erfinv.html)

## Examples

``` r
x <- as_mlx(seq(-3, 3, by = 0.5))
as.matrix(mlx_dnorm(x))
#>  [1] 0.004431848 0.017528297 0.053990964 0.129517585 0.241970733 0.352065325
#>  [7] 0.398942292 0.352065325 0.241970733 0.129517585 0.053990964 0.017528297
#> [13] 0.004431848
as.matrix(mlx_pnorm(x))
#>  [1] 0.001349896 0.006209671 0.022750139 0.066807210 0.158655256 0.308537543
#>  [7] 0.500000000 0.691462457 0.841344714 0.933192790 0.977249861 0.993790329
#> [13] 0.998650074

p <- as_mlx(c(0.025, 0.5, 0.975))
as.matrix(mlx_qnorm(p))
#> [1] -1.959964  0.000000  1.959964
```
