# Control Global Compilation Behavior

- `mlx_disable_compile()` prevents all compilation globally. Compiled
  functions will execute without optimization.

- `mlx_enable_compile()` enables compilation (overrides the
  `MLX_DISABLE_COMPILE` environment variable).

## Usage

``` r
mlx_disable_compile()

mlx_enable_compile()
```

## Value

Invisibly returns `NULL`.

## Details

These functions control whether MLX compilation is enabled globally.

These are useful for debugging (to check if compilation is causing
issues) or benchmarking (to measure compilation overhead vs speedup).

You can also disable compilation by setting the `MLX_DISABLE_COMPILE`
environment variable before loading the package.

## Examples

``` r
demo_fn <- mlx_compile(function(x) x + 1)
x <- mlx_rand_normal(c(4, 4))

# Disable compilation for debugging
mlx_disable_compile()
demo_fn(x)  # Runs without optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>           [,1]      [,2]      [,3]       [,4]
#> [1,] 0.3803226 2.7679772  0.470581  2.2987065
#> [2,] 0.7929736 0.3669322  2.980639  0.5120839
#> [3,] 0.4584070 0.7540243  1.218017  1.5265613
#> [4,] 2.3634415 1.6922596 -0.115180 -1.3950124

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>           [,1]      [,2]      [,3]       [,4]
#> [1,] 0.3803226 2.7679772  0.470581  2.2987065
#> [2,] 0.7929736 0.3669322  2.980639  0.5120839
#> [3,] 0.4584070 0.7540243  1.218017  1.5265613
#> [4,] 2.3634415 1.6922596 -0.115180 -1.3950124
```
