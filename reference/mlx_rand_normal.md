# Sample from a normal distribution on mlx arrays

Sample from a normal distribution on mlx arrays

## Usage

``` r
mlx_rand_normal(dim, mean = 0, sd = 1, dtype = c("float32", "float64"))
```

## Arguments

- dim:

  Integer vector specifying array dimensions (shape).

- mean:

  Mean of the normal distribution.

- sd:

  Standard deviation of the normal distribution.

- dtype:

  Data type string. Supported types include:

  - Floating point: `"float32"`, `"float64"`

  - Integer: `"int8"`, `"int16"`, `"int32"`, `"int64"`, `"uint8"`,
    `"uint16"`, `"uint32"`, `"uint64"`

  - Other: `"bool"`, `"complex64"`

  Not all functions support all types. See individual function
  documentation.

## Value

An mlx array with normally distributed entries.

## See also

[mlx.core.random.normal](https://ml-explore.github.io/mlx/build/html/python/random.html#mlx.core.random.normal)

## Examples

``` r
weights <- mlx_rand_normal(c(3, 3), mean = 0, sd = 0.1)
```
