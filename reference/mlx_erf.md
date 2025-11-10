# Error function and inverse error function

`mlx_erf()` computes the error function elementwise. `mlx_erfinv()`
computes the inverse error function elementwise.

## Usage

``` r
mlx_erf(x)

mlx_erfinv(x)
```

## Arguments

- x:

  An mlx array.

## Value

An mlx array with the result.

## See also

[mlx.core.erf](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.erf.html),
[mlx.core.erfinv](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.erfinv.html)

## Examples

``` r
x <- as_mlx(c(-1, 0, 1))
as.matrix(mlx_erf(x))
#> [1] -0.8427008  0.0000000  0.8427008
p <- as_mlx(c(-0.5, 0, 0.5))
as.matrix(mlx_erfinv(p))
#> [1] -0.4769363  0.0000000  0.4769363
```
