# Construct MLX vectors

`mlx_vector()` is a convenience around
[`mlx_array()`](https://hughjonesd.github.io/Rmlx/reference/mlx_array.md)
for 1-D payloads.

## Usage

``` r
mlx_vector(data, dtype = NULL, device = mlx_default_device())
```

## Arguments

- data:

  Atomic vector providing the elements (recycling is not allowed).

- dtype:

  Optional MLX dtype. Defaults to `"float32"` for numeric input,
  `"bool"` for logical, and `"complex64"` for complex.

- device:

  Execution target: provide `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  Defaults to the current
  [`mlx_default_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_default_device.md).

## Value

An `mlx` vector with `dim = length(data)`.
