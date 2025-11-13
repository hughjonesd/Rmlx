# Single training step helper

Single training step helper

## Usage

``` r
mlx_train_step(module, loss_fn, optimizer, ...)
```

## Arguments

- module:

  An `mlx_module`.

- loss_fn:

  Function of `module` and data returning an mlx scalar.

- optimizer:

  Optimizer object from
  [`mlx_optimizer_sgd()`](https://hughjonesd.github.io/Rmlx/reference/mlx_optimizer_sgd.md).

- ...:

  Additional data passed to `loss_fn`.

## Value

A list with the current loss.

## See also

[mlx.optimizers.Optimizer](https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.optimizers.Optimizer)

## Examples

``` r
set.seed(1)
model <- mlx_linear(2, 1, bias = FALSE)
opt <- mlx_optimizer_sgd(mlx_parameters(model), lr = 0.1)
data_x <- as_mlx(matrix(c(1, 2, 3, 4), 2, 2))
data_y <- as_mlx(matrix(c(5, 6), 2, 1))
loss_fn <- function(model, x, y) {
  pred <- model$forward(x)
  mean((pred - y)^2)
}
result <- mlx_train_step(model, loss_fn, opt, data_x, data_y)
```
