# SiLU (Swish) activation

Sigmoid Linear Unit, also known as Swish activation.

## Usage

``` r
mlx_silu()
```

## Value

An `mlx_module` applying SiLU activation.

## See also

[mlx.nn.SiLU](https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.SiLU)

## Examples

``` r
act <- mlx_silu()
x <- as_mlx(matrix(c(-2, -1, 0, 1, 2), 5, 1))
mlx_forward(act, x)
#> mlx array [5 x 1]
#>   dtype: float32
#>   device: gpu
#>   values:
#>            [,1]
#> [1,] -0.2384058
#> [2,] -0.2689414
#> [3,]  0.0000000
#> [4,]  0.7310586
#> [5,]  1.7615941
```
