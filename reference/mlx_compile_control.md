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
#>           [,1]        [,2]      [,3]       [,4]
#> [1,] 0.3099204  1.08073950 1.7282910 0.49279416
#> [2,] 1.9300368  0.86971730 0.9866772 1.82742786
#> [3,] 0.7768922 -0.03181434 2.5281243 0.06665576
#> [4,] 2.0742407  2.19521332 1.4847980 0.75764161

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>           [,1]        [,2]      [,3]       [,4]
#> [1,] 0.3099204  1.08073950 1.7282910 0.49279416
#> [2,] 1.9300368  0.86971730 0.9866772 1.82742786
#> [3,] 0.7768922 -0.03181434 2.5281243 0.06665576
#> [4,] 2.0742407  2.19521332 1.4847980 0.75764161
```
