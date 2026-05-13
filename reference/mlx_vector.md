# Construct MLX vectors

`mlx_vector()` is a convenience around
[`mlx_array()`](https://hughjonesd.github.io/Rmlx/reference/mlx_array.md)
for 1-D payloads.

## Usage

``` r
mlx_vector(data, dtype = NULL)
```

## Arguments

- data:

  Atomic vector providing the elements (recycling is not allowed).

- dtype:

  Data type string. Supported types include:

  - Floating point: `"float32"`, `"float64"`

  - Integer: `"int8"`, `"int16"`, `"int32"`, `"int64"`, `"uint8"`,
    `"uint16"`, `"uint32"`, `"uint64"`

  - Other: `"bool"`, `"complex64"`

  Not all functions support all types. See individual function
  documentation.

## Value

An `mlx` vector with `dim = length(data)`.
