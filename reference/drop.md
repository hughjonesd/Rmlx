# Drop singleton dimensions

`drop()` removes axes of length one. For base R objects this dispatches
to [`base::drop()`](https://rdrr.io/r/base/drop.html), while
`drop.mlx()` delegates to
[`mlx_squeeze()`](https://hughjonesd.github.io/Rmlx/reference/mlx_squeeze.md)
so that mlx arrays remain on the device.

## Usage

``` r
drop(x)

# Default S3 method
drop(x)

# S3 method for class 'mlx'
drop(x)
```

## Arguments

- x:

  Object to drop dimensions from.

## Value

An object with singleton dimensions removed. For mlx inputs the result
is another mlx array.

## See also

[`mlx_squeeze()`](https://hughjonesd.github.io/Rmlx/reference/mlx_squeeze.md),
[`base::drop()`](https://rdrr.io/r/base/drop.html)
