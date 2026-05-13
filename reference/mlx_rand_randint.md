# Sample random integers on mlx arrays

Generates random integers uniformly distributed over the interval \[low,
high).

## Usage

``` r
mlx_rand_randint(
  dim,
  low,
  high,
  dtype = c("int32", "int64", "uint32", "uint64")
)
```

## Arguments

- dim:

  Integer vector specifying array dimensions (shape).

- low:

  Lower bound (inclusive).

- high:

  Upper bound (exclusive).

- dtype:

  Data type string. Supported types include:

  - Floating point: `"float32"`, `"float64"`

  - Integer: `"int8"`, `"int16"`, `"int32"`, `"int64"`, `"uint8"`,
    `"uint16"`, `"uint32"`, `"uint64"`

  - Other: `"bool"`, `"complex64"`

  Not all functions support all types. See individual function
  documentation.

## Value

An mlx array of random integers.

## See also

[mlx.core.random.randint](https://ml-explore.github.io/mlx/build/html/python/random.html#mlx.core.random.randint)

## Examples

``` r
# Random integers from 0 to 9
samples <- mlx_rand_randint(c(3, 3), low = 0, high = 10)

# Random integers from -5 to 4
samples <- mlx_rand_randint(c(2, 5), low = -5, high = 5)
```
