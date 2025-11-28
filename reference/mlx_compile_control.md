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
#> [1,] 0.1763042 1.2618382 2.3658137 0.03286445
#> [2,] 1.3802003 2.3278213 0.8219709 0.99440157
#> [3,] 0.7637485 0.3149046 1.7321446 0.47248149
#> [4,] 2.2519217 1.6183856 1.3528159 2.42204475

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>           [,1]      [,2]      [,3]       [,4]
#> [1,] 0.1763042 1.2618382 2.3658137 0.03286445
#> [2,] 1.3802003 2.3278213 0.8219709 0.99440157
#> [3,] 0.7637485 0.3149046 1.7321446 0.47248149
#> [4,] 2.2519217 1.6183856 1.3528159 2.42204475
```
