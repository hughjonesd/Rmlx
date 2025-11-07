# Test if all elements of two arrays are close

Returns a boolean scalar indicating whether all elements of two arrays
are close within specified tolerances.

## Usage

``` r
mlx_allclose(
  a,
  b,
  rtol = 1e-05,
  atol = 1e-08,
  equal_nan = FALSE,
  device = mlx_default_device()
)
```

## Arguments

- a, b:

  MLX arrays or objects coercible to MLX arrays

- rtol:

  Relative tolerance (default: 1e-5)

- atol:

  Absolute tolerance (default: 1e-8)

- equal_nan:

  If `TRUE`, NaN values are considered equal (default: `FALSE`)

- device:

  Execution target: supply `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  Default:
  [`mlx_default_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_default_device.md).

## Value

An mlx array containing a single boolean value

## Details

Two values are considered close if:
`abs(a - b) <= (atol + rtol * abs(b))`

This function returns `TRUE` only if all elements are close. Supports
NumPy-style broadcasting.

## See also

[`mlx_isclose()`](https://hughjonesd.github.io/Rmlx/reference/mlx_isclose.md),
[`all.equal.mlx()`](https://hughjonesd.github.io/Rmlx/reference/all.equal.mlx.md),
[mlx.core.allclose](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.allclose.html)

## Examples

``` r
a <- as_mlx(c(1.0, 2.0, 3.0))
b <- as_mlx(c(1.0 + 1e-6, 2.0 + 1e-6, 3.0 + 1e-6))
as.logical(as.matrix(mlx_allclose(a, b)))  # TRUE
#> [1] TRUE
```
