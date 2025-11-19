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
#>          [,1]        [,2]      [,3]       [,4]
#> [1,] 2.444738  4.14161301 0.2336446  1.7561810
#> [2,] 1.720352 -0.02272809 1.8468609  0.6557125
#> [3,] 2.391363 -1.08388877 2.3748901 -1.1379673
#> [4,] 2.137570  0.93248814 3.4068222 -0.3091656

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>          [,1]        [,2]      [,3]       [,4]
#> [1,] 2.444738  4.14161301 0.2336446  1.7561810
#> [2,] 1.720352 -0.02272809 1.8468609  0.6557125
#> [3,] 2.391363 -1.08388877 2.3748901 -1.1379673
#> [4,] 2.137570  0.93248814 3.4068222 -0.3091656
```
