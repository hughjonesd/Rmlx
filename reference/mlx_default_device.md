# Get or set default MLX device

Get or set default MLX device

## Usage

``` r
mlx_default_device(value)
```

## Arguments

- value:

  New default device ("gpu" or "cpu"). If missing, returns current
  default.

## Value

Current default device (character)

## See also

[mlx.core.default_device](https://ml-explore.github.io/mlx/build/html/python/metal.html)

## Examples

``` r
mlx_default_device()  # Get current default
#> [1] "gpu"
mlx_default_device("cpu")  # Set to CPU
#> [1] "cpu"
mlx_default_device("gpu")  # Set back to GPU
#> [1] "gpu"
mlx_default_device()
#> [1] "gpu"
```
