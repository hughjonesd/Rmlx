# Binary cross-entropy loss

Computes binary cross-entropy loss between predictions and binary
targets.

## Usage

``` r
mlx_binary_cross_entropy(
  predictions,
  targets,
  reduction = c("mean", "sum", "none")
)
```

## Arguments

- predictions:

  Predicted probabilities as an mlx array (values in \[0,1\]).

- targets:

  Binary target values as an mlx array (0 or 1).

- reduction:

  Type of reduction: "mean" (default), "sum", or "none".

## Value

An mlx array containing the loss.

## See also

[mlx.nn.losses.binary_cross_entropy](https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.losses.binary_cross_entropy)

## Examples

``` r
preds <- mlx_matrix(c(0.9, 0.2, 0.8), 3, 1)
targets <- mlx_matrix(c(1, 0, 1), 3, 1)
mlx_binary_cross_entropy(preds, targets)
#> mlx array []
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 0.1838825
```
