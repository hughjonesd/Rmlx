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
#>            [,1]       [,2]      [,3]      [,4]
#> [1,]  2.1290219  1.1828244 2.2247148 1.4079838
#> [2,]  1.5679864 -0.3308146 0.8650374 2.7884245
#> [3,]  0.9127913 -0.8393149 1.8454821 0.4152774
#> [4,] -0.3249868  1.6668971 0.4085653 2.1740031

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>            [,1]       [,2]      [,3]      [,4]
#> [1,]  2.1290219  1.1828244 2.2247148 1.4079838
#> [2,]  1.5679864 -0.3308146 0.8650374 2.7884245
#> [3,]  0.9127913 -0.8393149 1.8454821 0.4152774
#> [4,] -0.3249868  1.6668971 0.4085653 2.1740031
```
