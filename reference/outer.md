# Outer product of two vectors

Outer product of two vectors

## Usage

``` r
outer(X, Y, FUN = "*", ...)

# S3 method for class 'mlx'
outer(X, Y, FUN = "*", ...)
```

## Arguments

- X, Y:

  Numeric vectors or mlx arrays.

- FUN:

  Function to apply (for default method).

- ...:

  Additional arguments passed to methods.

## Value

For mlx inputs, an mlx matrix. Otherwise delegates to
[`base::outer`](https://rdrr.io/r/base/outer.html).

## See also

[mlx.core.outer](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.outer.html)

## Examples

``` r
x <- as_mlx(c(1, 2, 3))
y <- as_mlx(c(4, 5))
outer(x, y)
#> mlx array [3 x 2]
#>   dtype: float32
#>   device: gpu
#>   values:
#>      [,1] [,2]
#> [1,]    4    5
#> [2,]    8   10
#> [3,]   12   15
```
