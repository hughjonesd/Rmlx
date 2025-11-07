# Generate raw random bits on MLX arrays

Generate raw random bits on MLX arrays

## Usage

``` r
mlx_key_bits(dim, width = 4L, key = NULL, device = mlx_default_device())
```

## Arguments

- dim:

  Integer vector specifying the array shape/dimensions.

- width:

  Number of bytes per element (default 4 = 32 bits). Must be positive.

- key:

  Optional `mlx` key array. If omitted, MLX's default generator is used.

- device:

  Execution target: provide `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  Defaults to the current
  [`mlx_default_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_default_device.md).

## Value

An `mlx` array of unsigned integers filled with random bits.

## See also

[mlx.core.random.bits](https://ml-explore.github.io/mlx/build/html/python/random.html#mlx.core.random.bits)

## Examples

``` r
k <- mlx_key(12)
raw_bits <- mlx_key_bits(c(4, 4), key = k)
```
