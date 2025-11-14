# Sample from a categorical distribution on mlx arrays

Samples indices from categorical distributions. Each row (or slice along
the specified axis) represents a separate categorical distribution over
classes.

## Usage

``` r
mlx_rand_categorical(logits, axis = -1L, num_samples = 1L)
```

## Arguments

- logits:

  A matrix or mlx array of log-probabilities. The values don't need to
  be normalized (the function applies softmax internally). For a single
  distribution over K classes, use a 1×K matrix. For multiple
  independent distributions, use an N×K matrix where each row is a
  distribution.

- axis:

  The axis (1-indexed, negatives count from the end) along which to
  sample. Default is -1L (last axis, typically the class dimension).

- num_samples:

  Number of samples to draw from each distribution.

## Value

An mlx array of integer indices (1-indexed) sampled from the categorical
distributions.

## See also

[mlx.core.random.categorical](https://ml-explore.github.io/mlx/build/html/python/random.html#mlx.core.random.categorical)

## Examples

``` r
# Single distribution over 3 classes
logits <- matrix(c(0.5, 0.2, 0.3), 1, 3)
samples <- mlx_rand_categorical(logits, num_samples = 10)

# Multiple distributions
logits <- matrix(c(1, 2, 3,
                   3, 2, 1), nrow = 2, byrow = TRUE)
samples <- mlx_rand_categorical(logits, num_samples = 5)
```
