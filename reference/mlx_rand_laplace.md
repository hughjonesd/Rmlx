# Sample from the Laplace distribution on mlx arrays

Sample from the Laplace distribution on mlx arrays

## Usage

``` r
mlx_rand_laplace(
  dim,
  loc = 0,
  scale = 1,
  dtype = c("float32", "float64"),
  device = mlx_default_device()
)
```

## Arguments

- dim:

  Integer vector specifying array dimensions (shape).

- loc:

  Location parameter (mean) of the Laplace distribution.

- scale:

  Scale parameter (diversity) of the Laplace distribution.

- dtype:

  Data type string. Supported types include:

  - Floating point: `"float32"`, `"float64"`

  - Integer: `"int8"`, `"int16"`, `"int32"`, `"int64"`, `"uint8"`,
    `"uint16"`, `"uint32"`, `"uint64"`

  - Other: `"bool"`, `"complex64"`

  `float64` arrays are CPU-only. Use `device = "cpu"` when creating or
  casting to `float64`, and cast back to `float32` before using the GPU.
  Not all functions support all types. See individual function
  documentation.

- device:

  Execution target: supply `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  By default, many functions use the
  [`mlx_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_device.md)
  of their first argument.

## Value

An mlx array with Laplace-distributed entries.

## See also

[mlx.core.random.laplace](https://ml-explore.github.io/mlx/build/html/python/random.html#mlx.core.random.laplace)

## Examples

``` r
samples <- mlx_rand_laplace(c(2, 3), loc = 0, scale = 1)
```
