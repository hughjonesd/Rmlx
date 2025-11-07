# Elementwise NaN and infinity predicates

`mlx_isnan()`, `mlx_isinf()`, and `mlx_isfinite()` wrap the
corresponding MLX elementwise predicates, returning boolean arrays.

## Usage

``` r
mlx_isnan(x)

mlx_isinf(x)

mlx_isfinite(x)
```

## Arguments

- x:

  An mlx array.

## Value

An mlx boolean array.

## See also

[mlx.core.isnan](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.isnan),
[mlx.core.isinf](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.isinf),
[mlx.core.isfinite](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.isfinite)
