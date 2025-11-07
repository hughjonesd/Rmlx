# Identity matrices on MLX devices

Identity matrices on MLX devices

## Usage

``` r
mlx_identity(n, dtype = c("float32", "float64"), device = mlx_default_device())
```

## Arguments

- n:

  Size of the square matrix.

- dtype:

  MLX dtype to use. One of `"float32"`, `"float64"`, `"int8"`,
  `"int16"`, `"int32"`, `"int64"`, `"uint8"`, `"uint16"`, `"uint32"`,
  `"uint64"`, `"bool"`, or `"complex64"`.

- device:

  Execution target: provide `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  Defaults to the current
  [`mlx_default_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_default_device.md).

## Value

An mlx identity matrix.

## See also

[mlx.core.identity](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.identity)

## Examples

``` r
I4 <- mlx_identity(4)
```
