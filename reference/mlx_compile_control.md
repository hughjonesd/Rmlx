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
#>             [,1]      [,2]      [,3]      [,4]
#> [1,]  1.17802536 4.5357862 0.2350672 1.1482550
#> [2,]  0.18212193 1.2885206 1.0295365 3.1430886
#> [3,] -0.99446368 1.3838103 1.6176388 0.6394063
#> [4,] -0.07818317 0.8037395 0.9481130 0.4193265

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>             [,1]      [,2]      [,3]      [,4]
#> [1,]  1.17802536 4.5357862 0.2350672 1.1482550
#> [2,]  0.18212193 1.2885206 1.0295365 3.1430886
#> [3,] -0.99446368 1.3838103 1.6176388 0.6394063
#> [4,] -0.07818317 0.8037395 0.9481130 0.4193265
```
