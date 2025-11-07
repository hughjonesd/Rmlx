# Construct MLX random number generator keys

`mlx_key()` provides access to MLX's stateless PRNG. Given a 64-bit seed
it returns a key that can be passed to other random helpers. Use
`mlx_key_split()` to derive multiple independent keys from an existing
key.

## Usage

``` r
mlx_key(seed)

mlx_key_split(key, num = 2L)
```

## Arguments

- seed:

  Integer or numeric seed (converted to unsigned 64-bit).

- key:

  An `mlx` key array returned by `mlx_key()`.

- num:

  Number of subkeys to produce (default 2L).

## Value

An `mlx` array holding the PRNG key.

A list of `num` `mlx` key arrays.

## See also

[mlx.core.random.key](https://ml-explore.github.io/mlx/build/html/python/random.html#mlx.core.random.key)

## Examples

``` r
k <- mlx_key(42)
subkeys <- mlx_key_split(k, num = 2)
```
