# Stochastic gradient descent optimizer

Stochastic gradient descent optimizer

## Usage

``` r
mlx_optimizer_sgd(params, lr = 0.01)
```

## Arguments

- params:

  List of parameters (from
  [`mlx_parameters()`](https://hughjonesd.github.io/Rmlx/reference/mlx_parameters.md)).

- lr:

  Learning rate.

## Value

An optimizer object with a [`step()`](https://rdrr.io/r/stats/step.html)
method.

## See also

[mlx.optimizers.SGD](https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.optimizers.SGD)

## Examples

``` r
set.seed(1)
model <- mlx_linear(2, 1, bias = FALSE)
opt <- mlx_optimizer_sgd(mlx_parameters(model), lr = 0.1)
```
