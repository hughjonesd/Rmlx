# Sample Bernoulli random variables on mlx arrays

Sample Bernoulli random variables on mlx arrays

## Usage

``` r
mlx_rand_bernoulli(dim, prob = 0.5, device = mlx_default_device())
```

## Arguments

- dim:

  Integer vector specifying array dimensions (shape).

- prob:

  Probability of a one.

- device:

  Execution target: supply `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  By default, many functions use the
  [`mlx_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_device.md)
  of their first argument.

## Value

An mlx boolean array.

## See also

[mlx.core.random.bernoulli](https://ml-explore.github.io/mlx/build/html/python/random.html#mlx.core.random.bernoulli)

## Examples

``` r
mask <- mlx_rand_bernoulli(c(4, 4), prob = 0.3)
```
