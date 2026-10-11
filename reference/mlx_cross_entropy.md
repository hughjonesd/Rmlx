# Cross-entropy loss

Computes cross-entropy loss for multi-class classification.

## Usage

``` r
mlx_cross_entropy(logits, targets, reduction = c("mean", "sum", "none"))
```

## Arguments

- logits:

  A matrix of unnormalized predictions (logits), with one row per sample
  and one column per class, as an mlx array or R matrix.

- targets:

  Target class indices as an mlx array or integer vector. Indices are
  1-based, with one non-missing index per sample.

- reduction:

  Type of reduction: "mean" (default), "sum", or "none".

## Value

A scalar mlx array for `"mean"` or `"sum"`, or a vector with one loss
per sample for `"none"`.

## Details

Gradients can be computed with respect to `logits`. Float64 arrays
require the CPU.

## See also

[mlx.nn.losses.cross_entropy](https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.losses.cross_entropy)

## Examples

``` r
# Logits for 3 samples, 4 classes
logits <- mlx_matrix(rnorm(12), 3, 4)
targets <- as_mlx(c(1, 3, 2))
mlx_cross_entropy(logits, targets)
#> mlx array []
#>   dtype: float32
#>   values:
#> [1] 2.319621
```
