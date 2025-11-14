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
#>           [,1]       [,2]       [,3]      [,4]
#> [1,] 0.1938273  0.9109979 0.03505105 0.6341144
#> [2,] 0.5349814 -0.1152490 1.77004170 1.3732346
#> [3,] 1.5549818  1.0423031 2.39143538 1.5479352
#> [4,] 0.4389477  1.8985431 1.84102011 0.6224436

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>           [,1]       [,2]       [,3]      [,4]
#> [1,] 0.1938273  0.9109979 0.03505105 0.6341144
#> [2,] 0.5349814 -0.1152490 1.77004170 1.3732346
#> [3,] 1.5549818  1.0423031 2.39143538 1.5479352
#> [4,] 0.4389477  1.8985431 1.84102011 0.6224436
```
