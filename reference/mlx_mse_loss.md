# Mean squared error loss

Computes the mean squared error between predictions and targets.

## Usage

``` r
mlx_mse_loss(predictions, targets, reduction = c("mean", "sum", "none"))
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

[mlx.nn.losses.mse_loss](https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.losses.mse_loss)

## Examples

``` r
preds <- as_mlx(matrix(c(1.5, 2.3, 0.8), 3, 1))
targets <- as_mlx(matrix(c(1, 2, 1), 3, 1))
mlx_mse_loss(preds, targets)
#> mlx array []
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 0.1266666
```
