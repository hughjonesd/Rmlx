# Replace NaN and infinite values with finite numbers

`mlx_nan_to_num()` mirrors
[`mlx.core.nan_to_num()`](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.nan_to_num),
filling non-finite entries with user-provided finite substitutes.

## Usage

``` r
mlx_nan_to_num(x, nan = 0, posinf = NULL, neginf = NULL)
```

## Arguments

- x:

  An mlx array.

- nan:

  Replacement for NaN values (default `0`). Use `NULL` to retain MLX's
  default.

- posinf:

  Optional replacement for positive infinity.

- neginf:

  Optional replacement for negative infinity.

## Value

An mlx array with non-finite values replaced.

## See also

[mlx.core.nan_to_num](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.nan_to_num)

## Examples

``` r
x <- as_mlx(c(-Inf, -1, NaN, 3, Inf))
mlx_nan_to_num(x, nan = 0, posinf = 10, neginf = -10)
#> mlx array [5]
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] -10  -1   0   3  10
```
