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

  Data type string. Supported types include:

  - Floating point: `"float32"`, `"float64"`

  - Integer: `"int8"`, `"int16"`, `"int32"`, `"int64"`, `"uint8"`,
    `"uint16"`, `"uint32"`, `"uint64"`

  - Other: `"bool"`, `"complex64"`

  `float64` arrays are CPU-only. Use `device = "cpu"` when creating or
  casting to `float64`, and cast back to `float32` before using the GPU.
  Not all functions support all types. See individual function
  documentation.

- device:

  Execution target: supply `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  By default, many functions use the
  [`mlx_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_device.md)
  of their first argument.

## Value

An mlx array with samples from the multivariate normal.

## Details

GPU execution is currently unavailable because the covariance
factorisation runs on the host.

## See also

[mlx.core.random.multivariate_normal](https://ml-explore.github.io/mlx/build/html/python/random.html#mlx.core.random.multivariate_normal)

## Examples

``` r
mean <- as_mlx(c(0, 0), device = "cpu")
cov <- as_mlx(matrix(c(1, 0, 0, 1), 2, 2), device = "cpu")
samples <- mlx_rand_multivariate_normal(10, mean, cov, device = "cpu")
```
