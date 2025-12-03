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
#>             [,1]      [,2]       [,3]       [,4]
#> [1,] -0.75174451 1.2505320 -0.9938128  1.8142486
#> [2,]  0.05059302 1.5656178  1.4014025  0.8746263
#> [3,]  0.60453129 0.1632969 -0.5127114 -0.8095087
#> [4,]  1.94334471 1.0383822 -0.1842272 -0.5092064

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>             [,1]      [,2]       [,3]       [,4]
#> [1,] -0.75174451 1.2505320 -0.9938128  1.8142486
#> [2,]  0.05059302 1.5656178  1.4014025  0.8746263
#> [3,]  0.60453129 0.1632969 -0.5127114 -0.8095087
#> [4,]  1.94334471 1.0383822 -0.1842272 -0.5092064
```
