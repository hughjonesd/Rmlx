# Cross-entropy loss

Computes cross-entropy loss for multi-class classification.

## Usage

``` r
mlx_cross_entropy(logits, targets, reduction = c("mean", "sum", "none"))
```

## Arguments

- logits:

  Unnormalized predictions (logits) as an mlx array.

- targets:

  Target class indices as an mlx array or integer vector.

- reduction:

  Type of reduction: "mean" (default), "sum", or "none".

## Value

An mlx array containing the loss.

## See also

[mlx.nn.losses.cross_entropy](https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.losses.cross_entropy)

## Examples

``` r
# Logits for 3 samples, 4 classes
logits <- as_mlx(matrix(rnorm(12), 3, 4))
targets <- as_mlx(c(1, 3, 2))  # 0-indexed class labels
mlx_cross_entropy(logits, targets)
#> mlx array []
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 1.120893
```
