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
#>           [,1]     [,2]       [,3]      [,4]
#> [1,] 0.9756016 2.128264 -0.2328833 0.4097518
#> [2,] 0.3434632 1.408249  2.0169249 0.7356906
#> [3,] 0.5307962 1.555236  1.7043992 0.8562694
#> [4,] 1.6838102 2.132762 -0.7550398 1.5705447

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   values:
#>           [,1]     [,2]       [,3]      [,4]
#> [1,] 0.9756016 2.128264 -0.2328833 0.4097518
#> [2,] 0.3434632 1.408249  2.0169249 0.7356906
#> [3,] 0.5307962 1.555236  1.7043992 0.8562694
#> [4,] 1.6838102 2.132762 -0.7550398 1.5705447
```
