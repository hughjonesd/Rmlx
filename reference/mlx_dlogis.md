# Logistic distribution functions

Compute density (`mlx_dlogis`), cumulative distribution (`mlx_plogis`),
and quantile (`mlx_qlogis`) functions for the logistic distribution
using MLX.

## Usage

``` r
mlx_dlogis(
  x,
  location = 0,
  scale = 1,
  log = FALSE,
  device = mlx_default_device()
)

mlx_plogis(x, location = 0, scale = 1, device = mlx_default_device())

mlx_qlogis(p, location = 0, scale = 1, device = mlx_default_device())
```

## Arguments

- x:

  Vector of quantiles (mlx array or coercible to mlx)

- location, scale:

  Location and scale parameters (default: 0, 1)

- log:

  If `TRUE`, return log density for `mlx_dlogis` (default: `FALSE`)

- device:

  Execution target: supply `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  Default:
  [`mlx_default_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_default_device.md).

- p:

  Vector of probabilities (mlx array or coercible to mlx)

## Value

An mlx array with the computed values

## Examples

``` r
x <- as_mlx(seq(-3, 3, by = 0.5))
as.matrix(mlx_dlogis(x))
#>  [1] 0.04517667 0.07010371 0.10499357 0.14914645 0.19661196 0.23500372
#>  [7] 0.25000000 0.23500372 0.19661194 0.14914647 0.10499358 0.07010371
#> [13] 0.04517666
as.matrix(mlx_plogis(x))
#>  [1] 0.04742587 0.07585818 0.11920292 0.18242553 0.26894143 0.37754068
#>  [7] 0.50000000 0.62245935 0.73105860 0.81757450 0.88079703 0.92414182
#> [13] 0.95257413

p <- as_mlx(c(0.25, 0.5, 0.75))
as.matrix(mlx_qlogis(p))
#> [1] -1.098612  0.000000  1.098612
```
