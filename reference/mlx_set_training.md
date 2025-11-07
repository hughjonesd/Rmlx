# Toggle training mode for MLX modules

`mlx_set_training()` switches modules between training and evaluation
modes. Layers that do not implement training-specific behaviour ignore
the call.

## Usage

``` r
mlx_set_training(module, mode = TRUE)
```

## Arguments

- module:

  An object inheriting from `mlx_module`.

- mode:

  Logical flag; `TRUE` for training mode, `FALSE` for evaluation.

## Value

The input module (invisibly).

## See also

<https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.Module.train>

## Examples

``` r
model <- mlx_sequential(mlx_linear(2, 4), mlx_dropout(0.5))
mlx_set_training(model, FALSE)
```
