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
#>             [,1]      [,2]     [,3]       [,4]
#> [1,]  0.53367567 0.6103438 1.341160  0.5716721
#> [2,]  2.34361267 2.2687988 2.290847  0.1263478
#> [3,] -0.07154274 1.0145687 1.999391  0.6262532
#> [4,]  1.03145611 1.4383534 1.863161 -0.2953743

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>             [,1]      [,2]     [,3]       [,4]
#> [1,]  0.53367567 0.6103438 1.341160  0.5716721
#> [2,]  2.34361267 2.2687988 2.290847  0.1263478
#> [3,] -0.07154274 1.0145687 1.999391  0.6262532
#> [4,]  1.03145611 1.4383534 1.863161 -0.2953743
```
