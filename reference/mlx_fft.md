# Fast Fourier transforms for MLX arrays

`mlx_fft()`, `mlx_fft2()`, and `mlx_fftn()` wrap the MLX FFT kernels
with R-friendly defaults. Inputs are converted with
[`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md) and
results are returned as `mlx` arrays.

## Usage

``` r
mlx_fft(x, axis = -1L, inverse = FALSE, device = NULL)

mlx_fft2(x, axes = c(-2L, -1L), inverse = FALSE, device = NULL)

mlx_fftn(x, axes = NULL, inverse = FALSE, device = NULL)
```

## Arguments

- x:

  Array-like object coercible to `mlx`.

- axis:

  Optional integer axis (1-indexed, negatives count from the end) for
  the one-dimensional transform.

- inverse:

  Logical flag; if `TRUE`, compute the inverse transform. The inverse is
  un-normalised to match base R's
  [`fft()`](https://hughjonesd.github.io/Rmlx/reference/fft.md), i.e.
  results are multiplied by the product of the transformed axis lengths.

- device:

  Target device or stream. Defaults to the input array's device (or
  [`mlx_default_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_default_device.md)
  for non-mlx inputs).

- axes:

  Optional integer vector of axes for the multi-dimensional transforms.
  When `NULL`, MLX uses all axes.

## Value

An `mlx` array containing complex frequency coefficients.

## See also

[`fft()`](https://hughjonesd.github.io/Rmlx/reference/fft.md),
[mlx.fft](https://ml-explore.github.io/mlx/build/html/python/fft.html)

## Examples

``` r
x <- as_mlx(c(1, 2, 3, 4))
mlx_fft(x)
#> mlx array [4]
#>   dtype: complex64
#>   device: gpu
#>   values:
#> [1] 10+0i -2+2i -2+0i -2-2i
mlx_fft(x, inverse = TRUE)
#> mlx array [4]
#>   dtype: complex64
#>   device: gpu
#>   values:
#> [1] 10+0i -2-2i -2+0i -2+2i
mat <- matrix(1:9, 3, 3)
mlx_fft2(as_mlx(mat))
#> mlx array [3 x 3]
#>   dtype: complex64
#>   device: gpu
#>   values:
#>                [,1]            [,2]            [,3]
#> [1,] 45.0+0.000000i -13.5+7.794229i -13.5-7.794229i
#> [2,] -4.5+2.598076i   0.0+0.000000i   0.0+0.000000i
#> [3,] -4.5-2.598076i   0.0+0.000000i   0.0+0.000000i
arr <- as_mlx(array(1:8, dim = c(2, 2, 2)))
mlx_fftn(arr)
#> mlx array [2 x 2 x 2]
#>   dtype: complex64
#>   device: gpu
#>   (8 elements, not shown)
```
