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
#>             [,1]         [,2]       [,3]     [,4]
#> [1,] 0.008996129 0.9659017324  0.5174259 1.037532
#> [2,] 2.198219299 0.6659241915 -0.3673137 1.963700
#> [3,] 1.064612389 0.0001615286  1.6659144 1.347574
#> [4,] 1.925594091 0.3953623772  0.2899656 2.031857

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   values:
#>             [,1]         [,2]       [,3]     [,4]
#> [1,] 0.008996129 0.9659017324  0.5174259 1.037532
#> [2,] 2.198219299 0.6659241915 -0.3673137 1.963700
#> [3,] 1.064612389 0.0001615286  1.6659144 1.347574
#> [4,] 1.925594091 0.3953623772  0.2899656 2.031857
```
