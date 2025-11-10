# Exponential distribution functions

Compute density (`mlx_dexp`), cumulative distribution (`mlx_pexp`), and
quantile (`mlx_qexp`) functions for the exponential distribution using
MLX.

## Usage

``` r
mlx_dexp(x, rate = 1, log = FALSE, device = mlx_default_device())

mlx_pexp(x, rate = 1, device = mlx_default_device())

mlx_qexp(p, rate = 1, device = mlx_default_device())
```

## Arguments

- x:

  Vector of quantiles (mlx array or coercible to mlx)

- rate:

  Rate parameter (default: 1)

- log:

  If `TRUE`, return log density for `mlx_dexp` (default: `FALSE`)

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
x <- as_mlx(seq(0, 5, by = 0.5))
as.matrix(mlx_dexp(x))
#>  [1] 1.000000000 0.606530666 0.367879450 0.223130152 0.135335281 0.082084998
#>  [7] 0.049787071 0.030197384 0.018315639 0.011108996 0.006737947
as.matrix(mlx_pexp(x))
#>  [1] 0.0000000 0.3934693 0.6321205 0.7768698 0.8646647 0.9179150 0.9502130
#>  [8] 0.9698026 0.9816844 0.9888910 0.9932621

p <- as_mlx(c(0.25, 0.5, 0.75))
as.matrix(mlx_qexp(p))
#> [1] 0.2876821 0.6931472 1.3862944
```
