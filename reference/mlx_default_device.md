# Get or set default MLX device

Get or set default MLX device

## Usage

``` r
mlx_default_device(value)
```

## Arguments

- value:

  New default device ("gpu" or "cpu"). If missing, returns the current
  default.

## Value

Current default device (character).

## See also

[mlx.core.default_device](https://ml-explore.github.io/mlx/build/html/python/metal.html)

## Examples

``` r
mlx_default_device()  # Get current default
#> [1] "gpu"
mlx_default_device("cpu")  # Set to CPU
#> [1] "cpu"
if (mlx_has_gpu()) {
  mlx_default_device("gpu")  # Set back to GPU
  mlx_default_device()
}
#> [1] "gpu"
mlx_default_device("cpu")
#> [1] "cpu"
```
