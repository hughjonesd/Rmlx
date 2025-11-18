# Sample random integers on mlx arrays

Generates random integers uniformly distributed over the interval \[low,
high).

## Usage

``` r
mlx_rand_randint(
  dim,
  low,
  high,
  dtype = c("int32", "int64", "uint32", "uint64"),
  device = mlx_default_device()
)
```

## Arguments

- dim:

  Integer vector specifying array dimensions (shape).

- low:

  Lower bound (inclusive).

- high:

  Upper bound (exclusive).

- dtype:

  Desired integer dtype ("int32", "int64", "uint32", "uint64").

- device:

  Execution target: supply `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  Defaults to the current
  [`mlx_default_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_default_device.md)
  unless noted otherwise (helpers that act on an existing array
  typically reuse that array's device or stream).

## Value

An mlx array of random integers.

## See also

[mlx.core.random.randint](https://ml-explore.github.io/mlx/build/html/python/random.html#mlx.core.random.randint)

## Examples

``` r
# Random integers from 0 to 9
samples <- mlx_rand_randint(c(3, 3), low = 0, high = 10)

# Random integers from -5 to 4
samples <- mlx_rand_randint(c(2, 5), low = -5, high = 5)
```
