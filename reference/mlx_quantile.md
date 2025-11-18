# Compute quantiles of MLX arrays

Calculates sample quantiles corresponding to given probabilities using
linear interpolation (R's type 7 quantiles, the default in
[`stats::quantile()`](https://rdrr.io/r/stats/quantile.html)). The S3
method `quantile.mlx()` provides an interface compatible with the
generic [`stats::quantile()`](https://rdrr.io/r/stats/quantile.html).

## Usage

``` r
mlx_quantile(
  x,
  probs,
  axis = NULL,
  drop = FALSE,
  device = mlx_default_device()
)

# S3 method for class 'mlx'
quantile(x, probs, ...)
```

## Arguments

- x:

  An mlx array, or an R array/matrix/vector that will be converted via
  [`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md).

- probs:

  Numeric vector of probabilities in \[0, 1\].

- axis:

  Optional integer axis (or vector of axes) along which to compute
  quantiles. When `NULL` (default), quantiles are computed over the
  entire flattened array.

- drop:

  Logical; when `TRUE` and computing quantiles along an axis with a
  single probability, removes the quantile dimension of length 1.
  Defaults to `FALSE` to match the behavior of other reduction
  functions.

- device:

  Execution target: supply `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  Defaults to the current
  [`mlx_default_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_default_device.md)
  unless noted otherwise (helpers that act on an existing array
  typically reuse that array's device or stream).

- ...:

  Additional arguments (currently ignored by `quantile.mlx()`).

## Value

An mlx array containing the requested quantiles. The shape depends on
`probs`, `axis`, and `drop`: when `axis = NULL`, returns a scalar for a
single probability or a vector for multiple probabilities. When `axis`
is specified, the quantile dimension replaces the reduced axis (e.g., a
`(3, 4)` matrix with `axis = 1` and 2 quantiles gives `(2, 4)`), unless
`drop = TRUE` with a single probability removes that dimension.

## Details

Uses type 7 quantiles (linear interpolation): for probability p and n
observations, the quantile is computed as:

- h = (n-1) \* p

- Interpolate between floor(h) and ceiling(h)

This matches the default behavior of
[`stats::quantile()`](https://rdrr.io/r/stats/quantile.html).

## See also

[`stats::quantile()`](https://rdrr.io/r/stats/quantile.html),
[mlx.core.sort](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.sort)

## Examples

``` r
x <- as_mlx(1:10)
as.numeric(mlx_quantile(x, 0.5))  # median
#> [1] 5.5
as.numeric(mlx_quantile(x, c(0.25, 0.5, 0.75)))  # quartiles
#> [1] 3.25 5.50 7.75

# S3 method:
quantile(x, probs = c(0, 0.25, 0.5, 0.75, 1))
#> mlx array []
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1]  1.00  3.25  5.50  7.75 10.00

# With axis parameter, quantile dimension replaces the reduced axis:
mat <- as_mlx(matrix(1:12, 3, 4))  # shape (3, 4)
result <- mlx_quantile(mat, c(0.25, 0.75), axis = 1)  # shape (2, 4)
result <- mlx_quantile(mat, 0.5, axis = 1)  # shape (1, 4)
result <- mlx_quantile(mat, 0.5, axis = 1, drop = TRUE)  # shape (4,)
```
