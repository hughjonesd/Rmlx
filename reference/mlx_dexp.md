# Exponential distribution functions

Compute density (`mlx_dexp`), cumulative distribution (`mlx_pexp`), and
quantile (`mlx_qexp`) functions for the exponential distribution using
MLX.

## Usage

``` r
mlx_dexp(x, rate = 1, log = FALSE)

mlx_pexp(x, rate = 1)

mlx_qexp(p, rate = 1)
```

## Arguments

- x:

  Vector of quantiles (mlx array or coercible to mlx)

- rate:

  Rate parameter (default: 1)

- log:

  If `TRUE`, return log density for `mlx_dexp` (default: `FALSE`)

- p:

  Vector of probabilities (mlx array or coercible to mlx)

## Value

An mlx array with the computed values.

## Examples

``` r
x <- as_mlx(seq(0, 5, by = 0.5))
mlx_dexp(x)
#> mlx array [11]
#>   dtype: float32
#>   values:
#>  [1] 1.000000000 0.606530666 0.367879450 0.223130181 0.135335281 0.082084998
#>  [7] 0.049787074 0.030197382 0.018315639 0.011108998 0.006737947
mlx_pexp(x)
#> mlx array [11]
#>   dtype: float32
#>   values:
#>  [1] 0.0000000 0.3934693 0.6321205 0.7768698 0.8646647 0.9179150 0.9502130
#>  [8] 0.9698026 0.9816844 0.9888910 0.9932621

p <- as_mlx(c(0.25, 0.5, 0.75))
mlx_qexp(p)
#> mlx array [3]
#>   dtype: float32
#>   values:
#> [1] 0.2876821 0.6931472 1.3862944
```
