# Get or set current MLX device

Get or set current MLX device

## Usage

``` r
mlx_device(value)
```

## Arguments

- value:

  New current device ("gpu" or "cpu"). If missing, returns the current
  device.

## Value

Current device (character).

## See also

[mlx.core.default_device](https://ml-explore.github.io/mlx/build/html/python/metal.html)

## Examples

``` r
mlx_device()  # Get current device
#> [1] "gpu"
mlx_device("cpu")  # Set to CPU
#> [1] "cpu"
if (mlx_has_gpu()) {
  mlx_device("gpu")  # Set back to GPU
  mlx_device()
}
#> [1] "gpu"
mlx_device("cpu")
#> [1] "cpu"
```
