# Sample from a uniform distribution on mlx arrays

Sample from a uniform distribution on mlx arrays

## Usage

``` r
mlx_rand_uniform(dim, min = 0, max = 1, dtype = c("float32", "float64"))
```

## Arguments

- dim:

  Integer vector specifying array dimensions (shape).

- min:

  Lower bound for the uniform distribution.

- max:

  Upper bound for the uniform distribution.

- dtype:

  Data type string. Supported types include:

  - Floating point: `"float32"`, `"float64"`

  - Integer: `"int8"`, `"int16"`, `"int32"`, `"int64"`, `"uint8"`,
    `"uint16"`, `"uint32"`, `"uint64"`

  - Other: `"bool"`, `"complex64"`

  Not all functions support all types. See individual function
  documentation.

## Value

An mlx array whose entries are sampled uniformly.

## See also

[mlx.core.random.uniform](https://ml-explore.github.io/mlx/build/html/python/random.html#mlx.core.random.uniform)

## Examples

``` r
noise <- mlx_rand_uniform(c(2, 2), min = -1, max = 1)
```
