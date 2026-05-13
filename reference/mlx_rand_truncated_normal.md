# Sample from a truncated normal distribution on mlx arrays

Sample from a truncated normal distribution on mlx arrays

## Usage

``` r
mlx_rand_truncated_normal(lower, upper, dim, dtype = c("float32", "float64"))
```

## Arguments

- lower:

  Lower bound of the truncated normal.

- upper:

  Upper bound of the truncated normal.

- dim:

  Integer vector specifying array dimensions (shape).

- dtype:

  Data type string. Supported types include:

  - Floating point: `"float32"`, `"float64"`

  - Integer: `"int8"`, `"int16"`, `"int32"`, `"int64"`, `"uint8"`,
    `"uint16"`, `"uint32"`, `"uint64"`

  - Other: `"bool"`, `"complex64"`

  Not all functions support all types. See individual function
  documentation.

## Value

An mlx array with truncated normally distributed entries.

## See also

[mlx.core.random.truncated_normal](https://ml-explore.github.io/mlx/build/html/python/random.html#mlx.core.random.truncated_normal)

## Examples

``` r
samples <- mlx_rand_truncated_normal(-1, 1, c(5, 5))
```
