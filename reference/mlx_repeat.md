# Repeat array elements

Repeat array elements

## Usage

``` r
mlx_repeat(x, repeats, axis = NULL)
```

## Arguments

- x:

  An mlx array.

- repeats:

  Number of repetitions.

- axis:

  Optional axis along which to repeat. When `NULL`, the array is
  flattened before repetition (matching NumPy semantics).

## Value

An mlx array with repeated values.

## See also

[mlx.core.repeat](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.repeat)

## Examples

``` r
x <- mlx_matrix(1:4, 2, 2)
mlx_repeat(x, repeats = 2, axis = 2)
#> mlx array [2 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1] [,2] [,3] [,4]
#> [1,]    1    1    3    3
#> [2,]    2    2    4    4
```
