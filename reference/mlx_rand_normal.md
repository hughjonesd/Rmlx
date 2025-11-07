# Sample from a normal distribution on mlx arrays

Sample from a normal distribution on mlx arrays

## Usage

``` r
mlx_rand_normal(
  dim,
  mean = 0,
  sd = 1,
  dtype = c("float32", "float64"),
  device = mlx_default_device()
)
```

## Arguments

- dim:

  Integer vector specifying the array shape/dimensions.

- mean:

  Mean of the normal distribution.

- sd:

  Standard deviation of the normal distribution.

- dtype:

  Desired MLX dtype ("float32" or "float64").

- device:

  Execution target: provide `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  Defaults to the current
  [`mlx_default_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_default_device.md).

## Value

An mlx array with normally distributed entries.

## See also

[mlx.core.random.normal](https://ml-explore.github.io/mlx/build/html/python/random.html#mlx.core.random.normal)

## Examples

``` r
weights <- mlx_rand_normal(c(3, 3), mean = 0, sd = 0.1)
```
