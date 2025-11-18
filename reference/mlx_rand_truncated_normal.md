# Sample from a truncated normal distribution on mlx arrays

Sample from a truncated normal distribution on mlx arrays

## Usage

``` r
mlx_rand_truncated_normal(
  lower,
  upper,
  dim,
  dtype = c("float32", "float64"),
  device = mlx_default_device()
)
```

## Arguments

- lower:

  Lower bound of the truncated normal.

- upper:

  Upper bound of the truncated normal.

- dim:

  Integer vector specifying array dimensions (shape).

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

An mlx array with truncated normally distributed entries.

## See also

[mlx.core.random.truncated_normal](https://ml-explore.github.io/mlx/build/html/python/random.html#mlx.core.random.truncated_normal)

## Examples

``` r
samples <- mlx_rand_truncated_normal(-1, 1, c(5, 5))
```
