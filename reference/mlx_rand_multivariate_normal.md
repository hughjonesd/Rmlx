# Sample from a multivariate normal distribution on mlx arrays

Sample from a multivariate normal distribution on mlx arrays

## Usage

``` r
mlx_rand_multivariate_normal(
  dim,
  mean,
  cov,
  dtype = c("float32", "float64"),
  device = "cpu"
)
```

## Arguments

- dim:

  Integer vector specifying array dimensions (shape).

- mean:

  An mlx array or vector for the mean.

- cov:

  An mlx array or matrix for the covariance.

- dtype:

  Desired MLX dtype ("float32" or "float64").

- device:

  Execution target: supply `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  Defaults to the current
  [`mlx_default_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_default_device.md)
  unless noted otherwise (helpers that act on an existing array
  typically reuse that array's device or stream).

## Value

An mlx array with samples from the multivariate normal.

## Details

Samples are generated on the CPU: GPU execution is currently unavailable
because the covariance factorisation runs on the host. Supply a CPU
stream (via
[`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md))
to integrate with asynchronous flows.

## See also

[mlx.core.random.multivariate_normal](https://ml-explore.github.io/mlx/build/html/python/random.html#mlx.core.random.multivariate_normal)

## Examples

``` r
mean <- as_mlx(c(0, 0), device = "cpu")
cov <- as_mlx(matrix(c(1, 0, 0, 1), 2, 2), device = "cpu")
samples <- mlx_rand_multivariate_normal(c(100, 2), mean, cov, device = "cpu")
```
