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
#>            [,1]      [,2]     [,3]      [,4]
#> [1,]  0.6930709 0.5206198 2.238049 0.9854901
#> [2,]  1.8084707 0.4558701 1.407002 1.1593838
#> [3,] -0.9718187 1.0057634 1.323719 0.9684238
#> [4,]  0.2228360 0.7096583 1.012656 1.0074680

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>            [,1]      [,2]     [,3]      [,4]
#> [1,]  0.6930709 0.5206198 2.238049 0.9854901
#> [2,]  1.8084707 0.4558701 1.407002 1.1593838
#> [3,] -0.9718187 1.0057634 1.323719 0.9684238
#> [4,]  0.2228360 0.7096583 1.012656 1.0074680
```
