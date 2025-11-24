# Shared arguments for MLX/base reduction helpers.

Shared arguments for MLX/base reduction helpers.

## Arguments

- x:

  An array or mlx array.

- na.rm:

  Logical; currently ignored for mlx arrays.

- dims:

  Leading dimensions treated as rows/cols (see
  [`base::rowSums()`](https://rdrr.io/r/base/colSums.html)).

- ...:

  Additional arguments forwarded to the base implementation.
