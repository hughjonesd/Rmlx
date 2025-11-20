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
#>              [,1]       [,2]       [,3]        [,4]
#> [1,]  2.186846733 0.34347302 -1.6210709  1.28968024
#> [2,] -0.003906846 2.19573998  1.8296020 -0.16183174
#> [3,]  0.063781798 0.06077749  0.5394566  1.14674234
#> [4,]  2.265684128 0.21183151  1.2114846  0.05953187

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>              [,1]       [,2]       [,3]        [,4]
#> [1,]  2.186846733 0.34347302 -1.6210709  1.28968024
#> [2,] -0.003906846 2.19573998  1.8296020 -0.16183174
#> [3,]  0.063781798 0.06077749  0.5394566  1.14674234
#> [4,]  2.265684128 0.21183151  1.2114846  0.05953187
```
