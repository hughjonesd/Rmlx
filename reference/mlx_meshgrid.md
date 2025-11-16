# Construct coordinate arrays from input vectors

`mlx_meshgrid()` mirrors
[`mlx.core.meshgrid()`](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.meshgrid),
returning coordinate arrays suitable for vectorised evaluation on MLX
devices.

## Usage

``` r
mlx_meshgrid(..., sparse = FALSE, indexing = c("xy", "ij"), device = NULL)
```

## Arguments

- ...:

  One or more arrays (or a single list) convertible via
  [`as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.md)
  representing coordinate vectors.

- sparse:

  Logical flag producing broadcast-friendly outputs when `TRUE`.

- indexing:

  Either `"xy"` (Cartesian) or `"ij"` (matrix) indexing.

- device:

  Execution target: supply `"gpu"`, `"cpu"`, or an `mlx_stream` created
  via
  [`mlx_new_stream()`](https://hughjonesd.github.io/Rmlx/reference/mlx_new_stream.md).
  Defaults to the current
  [`mlx_default_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_default_device.md)
  unless noted otherwise (helpers that act on an existing array
  typically reuse that array's device or stream).

## Value

A list of mlx arrays matching the number of inputs.

## See also

[mlx.core.meshgrid](https://ml-explore.github.io/mlx/build/html/python/array.html#mlx.core.meshgrid)

## Examples

``` r
xs <- as_mlx(1:3)
ys <- as_mlx(1:2)
grids <- mlx_meshgrid(xs, ys, indexing = "xy")
lapply(grids, as.matrix)
#> [[1]]
#>      [,1] [,2] [,3]
#> [1,]    1    2    3
#> [2,]    1    2    3
#> 
#> [[2]]
#>      [,1] [,2] [,3]
#> [1,]    1    1    1
#> [2,]    2    2    2
#> 
```
