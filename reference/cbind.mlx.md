# Column-bind mlx arrays

Column-bind mlx arrays

## Usage

``` r
# S3 method for class 'mlx'
cbind(..., deparse.level = 1)
```

## Arguments

- ...:

  Objects to bind. mlx arrays are kept in MLX; other inputs are coerced
  via
  [`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md).

- deparse.level:

  Compatibility argument accepted for S3 dispatch; ignored.

## Value

An mlx array stacked along the second axis.

## Details

Unlike base R's [`cbind()`](https://rdrr.io/r/base/cbind.html), this
function supports arrays with more than 2 dimensions and preserves all
dimensions except the second (which is summed across inputs). Base R's
[`cbind()`](https://rdrr.io/r/base/cbind.html) flattens
higher-dimensional arrays to matrices before binding.

## See also

[mlx.core.concatenate](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.concatenate)

## Examples

``` r
x <- as_mlx(matrix(1:4, 2, 2))
y <- as_mlx(matrix(5:8, 2, 2))
cbind(x, y)
#> mlx array [2 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1] [,2] [,3] [,4]
#> [1,]    1    3    5    7
#> [2,]    2    4    6    8
```
