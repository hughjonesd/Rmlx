# Common MLX parameters

Common MLX parameters

## Arguments

- device:

  Execution target for APIs that expose a one-off device or stream
  override. Supply `"gpu"`, `"cpu"`, or an `mlx_stream` created via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  Ordinary array operations use the current
  [`mlx_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_device.md)
  instead.

- dtype:

  Data type string. Supported types include:

  - Floating point: `"float32"`, `"float64"`

  - Integer: `"int8"`, `"int16"`, `"int32"`, `"int64"`, `"uint8"`,
    `"uint16"`, `"uint32"`, `"uint64"`

  - Other: `"bool"`, `"complex64"`

  Not all functions support all types. See individual function
  documentation.

- axis:

  Single axis (1-indexed). Supply a positive integer between 1 and the
  array rank. Use `NULL` when the helper interprets it as "all axes"
  (see individual docs).

- axes:

  Integer vector of axes (1-indexed). Supply positive integers between 1
  and the array rank. Many helpers interpret `NULL` to mean "all
  axes"—see the function details for specifics.

- drop:

  If `TRUE` (default), drop dimensions of length 1. If `FALSE`, retain
  all dimensions. Equivalent to `keepdims = TRUE` in underlying mlx
  functions.

- dim:

  Integer vector specifying array dimensions (shape).

- x:

  An mlx array, or an R array/matrix/vector that will be converted via
  [`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md).
