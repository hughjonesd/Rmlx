# Vector cross product with mlx arrays

Vector cross product with mlx arrays

## Usage

``` r
mlx_cross(a, b, axis = NULL)
```

## Arguments

- a, b:

  Input mlx arrays containing 3D vectors.

- axis:

  Axis along which to compute the cross product (1-indexed). Omit the
  argument to use the trailing dimension.

## Value

An mlx array of cross products.

## See also

[mlx.linalg.cross](https://ml-explore.github.io/mlx/build/html/python/linalg.html#mlx.linalg.cross)

## Examples

``` r
u <- as_mlx(c(1, 0, 0))
v <- as_mlx(c(0, 1, 0))
mlx_cross(u, v)
#> mlx array [3]
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 0 0 1
```
