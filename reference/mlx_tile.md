# Tile an array

Tile an array

## Usage

``` r
mlx_tile(x, reps)
```

## Arguments

- x:

  An mlx array.

- reps:

  Integer vector giving the number of repetitions for each axis.

## Value

An mlx array with tiled content.

## See also

[mlx.core.tile](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.tile)

## Examples

``` r
x <- as_mlx(matrix(1:4, 2, 2))
mlx_tile(x, reps = c(1, 2))
#> mlx array [2 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1] [,2] [,3] [,4]
#> [1,]    1    3    1    3
#> [2,]    2    4    2    4
```
