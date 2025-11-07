# Kronecker product dispatcher

Wrapper around
[`base::kronecker()`](https://rdrr.io/r/base/kronecker.html) that
enables S3 dispatch for `mlx` arrays while delegating to base R for all
other inputs.

Ensures the base `kronecker()` generic can dispatch on S3 `mlx` objects
when S4 dispatch is unavailable.

## Usage

``` r
kronecker(X, Y, FUN = "*", make.dimnames = FALSE, ...)

kronecker.default(X, Y, FUN = "*", make.dimnames = FALSE, ...)

# S4 method for class 'mlx,mlx'
kronecker(X, Y, FUN = "*", make.dimnames = FALSE, ...)

# S4 method for class 'mlx,ANY'
kronecker(X, Y, FUN = "*", make.dimnames = FALSE, ...)

# S4 method for class 'ANY,mlx'
kronecker(X, Y, FUN = "*", make.dimnames = FALSE, ...)

# S3 method for class 'mlx'
kronecker(X, Y, FUN = "*", ..., make.dimnames = FALSE)
```

## Arguments

- X:

  a vector or array.

- Y:

  a vector or array.

- FUN:

  Must be `'*'` (other functions are unsupported for MLX tensors).

- make.dimnames:

  logical: provide dimnames that are the product of the dimnames of `X`
  and `Y`.

- ...:

  Passed to maintain signature compatibility with base `kronecker()`.

## Value

An `mlx` array.
