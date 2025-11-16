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
#>            [,1]       [,2]      [,3]       [,4]
#> [1,]  0.6885960 -0.2165980 0.2565813 2.12118530
#> [2,]  1.7389933  2.8602319 2.6256876 0.04484504
#> [3,] -0.3415704  0.6531709 1.4265676 2.15548325
#> [4,]  1.7810299  0.9662346 0.8734691 0.88092554

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>            [,1]       [,2]      [,3]       [,4]
#> [1,]  0.6885960 -0.2165980 0.2565813 2.12118530
#> [2,]  1.7389933  2.8602319 2.6256876 0.04484504
#> [3,] -0.3415704  0.6531709 1.4265676 2.15548325
#> [4,]  1.7810299  0.9662346 0.8734691 0.88092554
```
