# Uniform distribution functions

Compute density (`mlx_dunif`), cumulative distribution (`mlx_punif`),
and quantile (`mlx_qunif`) functions for the uniform distribution using
MLX.

## Usage

``` r
mlx_dunif(x, min = 0, max = 1, log = FALSE)

mlx_punif(x, min = 0, max = 1)

mlx_qunif(p, min = 0, max = 1)
```

## Arguments

- x:

  Vector of quantiles (mlx array or coercible to mlx)

- min, max:

  Lower and upper limits of the distribution (default: 0, 1)

- log:

  If `TRUE`, return log density for `mlx_dunif` (default: `FALSE`)

- p:

  Vector of probabilities (mlx array or coercible to mlx)

## Value

An mlx array with the computed values.

## Examples

``` r
x <- as_mlx(seq(0, 1, by = 0.1))
mlx_dunif(x)
#> mlx array [11]
#>   dtype: float32
#>   values:
#>  [1] 1 1 1 1 1 1 1 1 1 1 1
mlx_punif(x)
#> mlx array [11]
#>   dtype: float32
#>   values:
#>  [1] 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

p <- as_mlx(c(0.25, 0.5, 0.75))
mlx_qunif(p)
#> mlx array [3]
#>   dtype: float32
#>   values:
#> [1] 0.25 0.50 0.75
```
