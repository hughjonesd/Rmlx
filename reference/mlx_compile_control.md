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
#>           [,1]       [,2]        [,3]     [,4]
#> [1,] 0.8339967  0.8393820  0.07618040 2.323762
#> [2,] 2.2028151  1.6226028  1.43680000 1.950190
#> [3,] 0.6422796 -0.2887880 -0.31888711 1.449617
#> [4,] 1.3406274 -0.6354238  0.07906324 1.434310

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>           [,1]       [,2]        [,3]     [,4]
#> [1,] 0.8339967  0.8393820  0.07618040 2.323762
#> [2,] 2.2028151  1.6226028  1.43680000 1.950190
#> [3,] 0.6422796 -0.2887880 -0.31888711 1.449617
#> [4,] 1.3406274 -0.6354238  0.07906324 1.434310
```
