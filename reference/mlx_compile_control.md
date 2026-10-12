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
#>   values:
#>            [,1]      [,2]        [,3]        [,4]
#> [1,]  3.3055701 1.3146876  1.84718704  0.76180065
#> [2,]  0.5648873 0.3984675 -0.02247524  0.39954352
#> [3,] -0.3486581 0.6734124  1.06721449 -0.07758284
#> [4,]  1.8784058 0.6559527  1.21627724  1.27350771

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   values:
#>            [,1]      [,2]        [,3]        [,4]
#> [1,]  3.3055701 1.3146876  1.84718704  0.76180065
#> [2,]  0.5648873 0.3984675 -0.02247524  0.39954352
#> [3,] -0.3486581 0.6734124  1.06721449 -0.07758284
#> [4,]  1.8784058 0.6559527  1.21627724  1.27350771
```
