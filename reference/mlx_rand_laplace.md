# Sample from the Laplace distribution on mlx arrays

Sample from the Laplace distribution on mlx arrays

## Usage

``` r
mlx_rand_laplace(
  dim,
  loc = 0,
  scale = 1,
  dtype = c("float32", "float64"),
  device = mlx_default_device()
)
```

## Arguments

- dim:

  Integer vector specifying array dimensions (shape).

- loc:

  Location parameter (mean) of the Laplace distribution.

- scale:

  Scale parameter (diversity) of the Laplace distribution.

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

An mlx array with Laplace-distributed entries.

## See also

[mlx.core.random.laplace](https://ml-explore.github.io/mlx/build/html/python/random.html#mlx.core.random.laplace)

## Examples

``` r
samples <- mlx_rand_laplace(c(2, 3), loc = 0, scale = 1)
```
