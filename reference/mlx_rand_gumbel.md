# Sample from the Gumbel distribution on mlx arrays

Sample from the Gumbel distribution on mlx arrays

## Usage

``` r
mlx_rand_gumbel(
  dim,
  dtype = c("float32", "float64"),
  device = mlx_default_device()
)
```

## Arguments

- dim:

  Integer vector specifying the array shape/dimensions.

- dtype:

  Desired MLX dtype ("float32" or "float64").

- device:

  Execution target: provide `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  Defaults to the current
  [`mlx_default_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_default_device.md).

## Value

An mlx array with Gumbel-distributed entries.

## See also

[mlx.core.random.gumbel](https://ml-explore.github.io/mlx/build/html/python/random.html#mlx.core.random.gumbel)

## Examples

``` r
samples <- mlx_rand_gumbel(c(2, 3))
```
