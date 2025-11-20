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
#>          [,1]       [,2]     [,3]        [,4]
#> [1,] 0.975535  2.1396966 2.506315  0.88885272
#> [2,] 2.138525  1.8176253 0.212326  1.95947671
#> [3,] 1.692814 -0.9639629 1.928394  0.29002893
#> [4,] 2.146777  0.3794861 3.233627 -0.07618582

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>          [,1]       [,2]     [,3]        [,4]
#> [1,] 0.975535  2.1396966 2.506315  0.88885272
#> [2,] 2.138525  1.8176253 0.212326  1.95947671
#> [3,] 1.692814 -0.9639629 1.928394  0.29002893
#> [4,] 2.146777  0.3794861 3.233627 -0.07618582
```
