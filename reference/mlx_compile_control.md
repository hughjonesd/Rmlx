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
#>          [,1]     [,2]      [,3]      [,4]
#> [1,] 1.044275 1.372365 0.8140268 1.9273951
#> [2,] 1.166136 1.158033 1.9764795 0.2834499
#> [3,] 2.608934 2.172826 1.9022220 1.0565207
#> [4,] 1.207778 1.041538 2.3611863 2.1656251

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>          [,1]     [,2]      [,3]      [,4]
#> [1,] 1.044275 1.372365 0.8140268 1.9273951
#> [2,] 1.166136 1.158033 1.9764795 0.2834499
#> [3,] 2.608934 2.172826 1.9022220 1.0565207
#> [4,] 1.207778 1.041538 2.3611863 2.1656251
```
