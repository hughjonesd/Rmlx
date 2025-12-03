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
#>             [,1]      [,2]     [,3]      [,4]
#> [1,] -0.95327425 0.1137243 1.487140 2.5666857
#> [2,]  1.74173522 1.0060645 1.013033 1.5454886
#> [3,] -0.57081926 1.3831158 1.469533 0.8084806
#> [4,]  0.06674701 0.4608350 2.402277 0.8230038

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>             [,1]      [,2]     [,3]      [,4]
#> [1,] -0.95327425 0.1137243 1.487140 2.5666857
#> [2,]  1.74173522 1.0060645 1.013033 1.5454886
#> [3,] -0.57081926 1.3831158 1.469533 0.8084806
#> [4,]  0.06674701 0.4608350 2.402277 0.8230038
```
