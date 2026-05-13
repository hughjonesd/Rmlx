# Sample Bernoulli random variables on mlx arrays

Sample Bernoulli random variables on mlx arrays

## Usage

``` r
mlx_rand_bernoulli(dim, prob = 0.5)
```

## Arguments

- dim:

  Integer vector specifying array dimensions (shape).

- prob:

  Probability of a one.

## Value

An mlx boolean array.

## See also

[mlx.core.random.bernoulli](https://ml-explore.github.io/mlx/build/html/python/random.html#mlx.core.random.bernoulli)

## Examples

``` r
mask <- mlx_rand_bernoulli(c(4, 4), prob = 0.3)
```
