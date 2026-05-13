# Construct MLX scalars

Construct MLX scalars

## Usage

``` r
mlx_scalar(value, dtype = NULL)
```

## Arguments

- value:

  Single value (numeric, logical, or complex).

- dtype:

  Data type string. Supported types include:

  - Floating point: `"float32"`, `"float64"`

  - Integer: `"int8"`, `"int16"`, `"int32"`, `"int64"`, `"uint8"`,
    `"uint16"`, `"uint32"`, `"uint64"`

  - Other: `"bool"`, `"complex64"`

  Not all functions support all types. See individual function
  documentation.

## Value

A dimensionless `mlx` scalar.
