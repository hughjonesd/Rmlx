# Complex-valued helpers for mlx arrays

`mlx_real()`, `mlx_imag()`, and `mlx_conjugate()` expose MLX's complex
helpers to extract the real part, imaginary part, or complex conjugate
of an `mlx` array. Corresponding S3 methods for
[`Re()`](https://rdrr.io/r/base/complex.html),
[`Im()`](https://rdrr.io/r/base/complex.html), and
[`Conj()`](https://rdrr.io/r/base/complex.html) are also provided.

## Usage

``` r
mlx_real(x)

mlx_imag(x)

mlx_conjugate(x)
```

## Arguments

- x:

  An mlx array.

## Value

An `mlx` array containing the requested component.

## See also

[mlx.core.array](https://ml-explore.github.io/mlx/build/html/python/array.html#complex-helpers)

## Examples

``` r
z <- as_mlx(1:4 + 1i * (4:1))
mlx_real(z)
#> mlx array [4]
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 1 2 3 4
Im(z)
#> mlx array [4]
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 4 3 2 1
```
