# Cast an mlx array

`mlx_cast()` converts an mlx array to a different dtype and/or device
without changing its shape.

## Usage

``` r
mlx_cast(x, dtype = NULL, device = x$device)
```

## Arguments

- x:

  An mlx array.

- dtype:

  Target dtype string. Defaults to the array's current dtype.

- device:

  Target device string. Defaults to the array's current device.

## Value

An mlx array with the requested dtype and device.

## See also

[`mlx_dtype()`](https://hughjonesd.github.io/Rmlx/reference/mlx_dtype.md),
[`mlx_device()`](https://hughjonesd.github.io/Rmlx/reference/mlx_device.md)

## Examples

``` r
x <- mlx_vector(1:3, dtype = "int32")
mlx_cast(x, dtype = "float32")
#> mlx array [3]
#>   dtype: float32
#>   device: gpu
#>   values:
#> [1] 1 2 3
```
