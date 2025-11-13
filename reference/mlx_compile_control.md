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
#>            [,1]      [,2]      [,3]        [,4]
#> [1,] -0.7065232 1.8419173 2.0950334 -0.09962797
#> [2,]  0.8837854 0.7878885 0.9338580 -0.42792308
#> [3,]  1.6274271 1.0530596 1.7001524  1.97219622
#> [4,]  0.9288416 0.6823525 0.9838506 -0.48673761

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>            [,1]      [,2]      [,3]        [,4]
#> [1,] -0.7065232 1.8419173 2.0950334 -0.09962797
#> [2,]  0.8837854 0.7878885 0.9338580 -0.42792308
#> [3,]  1.6274271 1.0530596 1.7001524  1.97219622
#> [4,]  0.9288416 0.6823525 0.9838506 -0.48673761
```
