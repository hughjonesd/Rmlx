# Force evaluation of lazy MLX operations

Force evaluation of lazy MLX operations

## Usage

``` r
mlx_eval(x)
```

## Arguments

- x:

  An mlx array.

## Value

The input object (invisibly)

## See also

[mlx.core.eval](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.eval)

## Examples

``` r
x <- as_mlx(matrix(1:4, 2, 2))
mlx_eval(x)
```
