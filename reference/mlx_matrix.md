# Construct MLX matrices efficiently

`mlx_matrix()` wraps
[`mlx_array()`](https://hughjonesd.github.io/Rmlx/reference/mlx_array.md)
for the common 2-D case. It accepts the same style arguments as
[`base::matrix()`](https://rdrr.io/r/base/matrix.html) but without
recycling, so mistakes surface early. Supply `nrow` or `ncol` (the other
may be inferred from `length(data)`).

## Usage

``` r
mlx_matrix(
  data,
  nrow = NULL,
  ncol = NULL,
  byrow = FALSE,
  dtype = NULL,
  device = mlx_default_device()
)
```

## Arguments

- data:

  Numeric, logical, or complex vector supplying the payload. Any
  dimension attributes are ignored; pass `dim` explicitly.

- nrow, ncol:

  Matrix dimensions (positive integers).

- byrow:

  Logical; if `TRUE`, fill by rows (same semantics as
  [`base::matrix()`](https://rdrr.io/r/base/matrix.html)).

- dtype:

  Optional MLX dtype. Defaults to `"float32"` for numeric input,
  `"bool"` for logical, and `"complex64"` for complex.

- device:

  Execution target: supply `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  Defaults to the current
  [`mlx_default_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_default_device.md)
  unless noted otherwise (helpers that act on an existing array
  typically reuse that array's device or stream).

## Value

An `mlx` matrix with `dim = c(nrow, ncol)`.

## Examples

``` r
mx <- mlx_matrix(1:6, nrow = 2, ncol = 3, byrow = TRUE)
as.matrix(mx)
#>      [,1] [,2] [,3]
#> [1,]    1    2    3
#> [2,]    4    5    6
```
