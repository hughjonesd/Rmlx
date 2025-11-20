# Mean of MLX array elements

Mean of MLX array elements

## Usage

``` r
# S3 method for class 'mlx'
mean(x, ...)
```

## Arguments

- x:

  An mlx array.

- ...:

  Additional arguments (ignored)

## Value

An mlx scalar.

## See also

[mlx.core.mean](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.mean)

## Examples

``` r
x <- mlx_matrix(1:4, 2, 2)
mean(x)
#> mlx array []
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 2.5
```
