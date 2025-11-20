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
#>            [,1]      [,2]       [,3]       [,4]
#> [1,]  1.8779845  1.890351  1.7902044 0.07098597
#> [2,]  0.0677045  1.835276 -0.2794075 0.77807963
#> [3,] -0.1579595 -1.801940  1.5748093 2.05971575
#> [4,] -1.7941704  0.783284  2.3636355 1.90229571

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>            [,1]      [,2]       [,3]       [,4]
#> [1,]  1.8779845  1.890351  1.7902044 0.07098597
#> [2,]  0.0677045  1.835276 -0.2794075 0.77807963
#> [3,] -0.1579595 -1.801940  1.5748093 2.05971575
#> [4,] -1.7941704  0.783284  2.3636355 1.90229571
```
