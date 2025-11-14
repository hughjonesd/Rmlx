# Dropout layer

Dropout layer

## Usage

``` r
mlx_dropout(p = 0.5)
```

## Arguments

- p:

  Probability of dropping an element (default: 0.5).

## Value

An `mlx_module` applying dropout during training.

## See also

[mlx.nn.Dropout](https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.Dropout)

## Examples

``` r
set.seed(1)
dropout <- mlx_dropout(p = 0.3)
x <- as_mlx(matrix(1:12, 3, 4))
mlx_forward(dropout, x)
#> mlx array [3 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>          [,1]     [,2]     [,3]     [,4]
#> [1,] 1.428571 5.714286 10.00000  0.00000
#> [2,] 2.857143 7.142857 11.42857 15.71429
#> [3,] 4.285714 8.571428 12.85714 17.14286
```
