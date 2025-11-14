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
#>            [,1]     [,2]       [,3]      [,4]
#> [1,] -1.0675590 1.012389  0.7224102 1.0176528
#> [2,]  0.9957128 1.631558  2.6986275 2.4297702
#> [3,]  1.1056288 1.568879 -1.3959563 3.0813463
#> [4,]  1.0490702 1.636941  0.3618059 0.8544121

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>            [,1]     [,2]       [,3]      [,4]
#> [1,] -1.0675590 1.012389  0.7224102 1.0176528
#> [2,]  0.9957128 1.631558  2.6986275 2.4297702
#> [3,]  1.1056288 1.568879 -1.3959563 3.0813463
#> [4,]  1.0490702 1.636941  0.3618059 0.8544121
```
