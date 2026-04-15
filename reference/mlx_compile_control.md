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
#>            [,1]       [,2]       [,3]      [,4]
#> [1,] -0.2464941  0.7562467 -0.6432542 1.0683783
#> [2,]  0.7614922 -0.3034768  1.7601178 0.9725761
#> [3,] -0.7707309  2.2094772  0.2324389 0.2569357
#> [4,]  2.0198541  1.1746192  0.9856025 0.6742412

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>            [,1]       [,2]       [,3]      [,4]
#> [1,] -0.2464941  0.7562467 -0.6432542 1.0683783
#> [2,]  0.7614922 -0.3034768  1.7601178 0.9725761
#> [3,] -0.7707309  2.2094772  0.2324389 0.2569357
#> [4,]  2.0198541  1.1746192  0.9856025 0.6742412
```
