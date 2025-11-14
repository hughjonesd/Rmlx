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
#>            [,1]      [,2]       [,3]     [,4]
#> [1,]  0.5549926 3.0320823  2.5363152 1.560285
#> [2,]  2.7635119 0.5196974  2.2632644 2.509062
#> [3,] -0.1001363 0.7940638 -0.3002102 1.999992
#> [4,] -1.6089690 2.3754697 -0.1510614 1.315734

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>            [,1]      [,2]       [,3]     [,4]
#> [1,]  0.5549926 3.0320823  2.5363152 1.560285
#> [2,]  2.7635119 0.5196974  2.2632644 2.509062
#> [3,] -0.1001363 0.7940638 -0.3002102 1.999992
#> [4,] -1.6089690 2.3754697 -0.1510614 1.315734
```
