# Create a learnable linear transformation

Create a learnable linear transformation

## Usage

``` r
mlx_linear(in_features, out_features, bias = TRUE)
```

## Arguments

- in_features:

  Number of input features.

- out_features:

  Number of output features.

- bias:

  Should a bias term be included?

## Value

An object of class `mlx_module`.

## See also

[mlx.nn.Linear](https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.Linear)

## Examples

``` r
set.seed(1)
layer <- mlx_linear(3, 2)
x <- mlx_matrix(1:6, 2, 3)
mlx_forward(layer, x)
#> mlx array [2 x 2]
#>   dtype: float32
#>   values:
#>           [,1]       [,2]
#> [1,] -3.473105 -1.2398810
#> [2,] -4.516946 -0.3382072
```
