# Generate raw random bits on MLX arrays

Generate raw random bits on MLX arrays

## Usage

``` r
mlx_key_bits(dim, width = 4L, key = NULL)
```

## Arguments

- dim:

  Integer vector specifying array dimensions (shape).

- width:

  Number of bytes per element (default 4 = 32 bits). Must be positive.

- key:

  Optional `mlx` key array. If omitted, MLX's default generator is used.

## Value

An `mlx` array of unsigned integers filled with random bits.

## See also

[mlx.core.random.bits](https://ml-explore.github.io/mlx/build/html/python/random.html#mlx.core.random.bits)

## Examples

``` r
k <- mlx_key(12)
raw_bits <- mlx_key_bits(c(4, 4), key = k)
```
