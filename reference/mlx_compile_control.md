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
#>           [,1]       [,2]       [,3]     [,4]
#> [1,] 0.6503716  3.6918910  0.4012863 2.000641
#> [2,] 0.7067928  0.8697210 -0.2722901 1.501650
#> [3,] 0.3778972  2.7711577  0.3416377 1.731067
#> [4,] 1.2161231 -0.4449017  1.4973130 1.606025

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>           [,1]       [,2]       [,3]     [,4]
#> [1,] 0.6503716  3.6918910  0.4012863 2.000641
#> [2,] 0.7067928  0.8697210 -0.2722901 1.501650
#> [3,] 0.3778972  2.7711577  0.3416377 1.731067
#> [4,] 1.2161231 -0.4449017  1.4973130 1.606025
```
