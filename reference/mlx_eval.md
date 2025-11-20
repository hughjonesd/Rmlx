# Force evaluation of an MLX operations

By default MLX computations are lazy. `mlx_eval(x)` forces the
computations behind `x` to run. You can do the same by calling (e.g.)
[as.matrix(x)](https://hughjonesd.github.io/Rmlx/reference/as.matrix.mlx.md).

## Usage

``` r
mlx_eval(x)
```

## Arguments

- x:

  An mlx array.

## Value

The input object, invisibly.

## See also

[mlx.core.eval](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.eval)

## Examples

``` r
system.time(x <- mlx_rand_normal(1e7))
#>    user  system elapsed 
#>       0       0       0 
system.time(mlx_eval(x))
#>    user  system elapsed 
#>   0.002   0.004   0.040 
```
