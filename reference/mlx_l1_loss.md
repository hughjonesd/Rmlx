# L1 loss (Mean Absolute Error)

Computes the mean absolute error between predictions and targets.

## Usage

``` r
mlx_l1_loss(predictions, targets, reduction = c("mean", "sum", "none"))
```

## Arguments

- predictions:

  Predicted values as an mlx array.

- targets:

  Target values as an mlx array.

- reduction:

  Type of reduction: "mean" (default), "sum", or "none".

## Value

An mlx array containing the loss.

## See also

[mlx.nn.losses.l1_loss](https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.losses.l1_loss)

## Examples

``` r
preds <- mlx_matrix(c(1.5, 2.3, 0.8), 3, 1)
targets <- mlx_matrix(c(1, 2, 1), 3, 1)
mlx_l1_loss(preds, targets)
#> mlx array []
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 0.3333333
```
