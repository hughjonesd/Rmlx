# Fast Fourier Transform

Extends [`stats::fft()`](https://rdrr.io/r/stats/fft.html) to work with
mlx objects while delegating to the standard R implementation for other
inputs.

## Usage

``` r
fft(z, inverse = FALSE, ...)
```

## Arguments

- z:

  Input to transform. May be a numeric, complex, or mlx object.

- inverse:

  Logical flag; if `TRUE` compute the inverse transform.

- ...:

  Passed through to the default method.

## Value

For mlx inputs, an mlx object containing complex frequency coefficients;
otherwise the base R result.

## See also

[`stats::fft()`](https://rdrr.io/r/stats/fft.html),
[`mlx_fft()`](https://hughjonesd.github.io/Rmlx/reference/mlx_fft.md),
[`mlx_fft2()`](https://hughjonesd.github.io/Rmlx/reference/mlx_fft.md),
[`mlx_fftn()`](https://hughjonesd.github.io/Rmlx/reference/mlx_fft.md),
[mlx.core.fft.fft](https://ml-explore.github.io/mlx/build/html/python/fft.html#mlx.core.fft.fft)

## Examples

``` r
z <- as_mlx(c(1, 2, 3, 4))
fft(z)
#> mlx array []
#>   dtype: complex64
#>   device: gpu
#>   values:
#> [1] 10+0i -2+2i -2+0i -2-2i
fft(z, inverse = TRUE)
#> mlx array []
#>   dtype: complex64
#>   device: gpu
#>   values:
#> [1] 10+0i -2-2i -2+0i -2+2i
```
