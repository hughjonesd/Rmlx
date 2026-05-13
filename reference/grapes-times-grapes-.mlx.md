# Matrix multiplication for MLX arrays

Both operands must be 2D matrices; vectors are not auto-promoted (unlike
base R).

## Usage

``` r
# S3 method for class 'mlx'
x %*% y
```

## Arguments

- x, y:

  numeric or complex matrices or vectors.

## Value

An mlx object.

## See also

[mlx.core.matmul](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.matmul)

## Examples

``` r
x <- mlx_matrix(1:6, 2, 3)
y <- mlx_matrix(1:6, 3, 2)
x %*% y
#> mlx array [2 x 2]
#>   dtype: float32
#>   values:
#>      [,1] [,2]
#> [1,]   22   49
#> [2,]   28   64
```
