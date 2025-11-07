# Sample from a uniform distribution on mlx arrays

Sample from a uniform distribution on mlx arrays

## Usage

``` r
mlx_rand_uniform(
  dim,
  min = 0,
  max = 1,
  dtype = c("float32", "float64"),
  device = mlx_default_device()
)
```

## Arguments

- dim:

  Integer vector specifying the array shape/dimensions.

- min:

  Lower bound for the uniform distribution.

- max:

  Upper bound for the uniform distribution.

- dtype:

  Desired MLX dtype ("float32" or "float64").

- device:

  Execution target: provide `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  Defaults to the current
  [`mlx_default_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_default_device.md).

## Value

An mlx array whose entries are sampled uniformly.

## See also

[mlx.core.random.uniform](https://ml-explore.github.io/mlx/build/html/python/random.html#mlx.core.random.uniform)

## Examples

``` r
noise <- mlx_rand_uniform(c(2, 2), min = -1, max = 1)
```
