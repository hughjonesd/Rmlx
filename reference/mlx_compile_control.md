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
#>           [,1]     [,2]       [,3]       [,4]
#> [1,] -0.422219 3.018452  0.3255044  1.9746192
#> [2,]  1.570685 1.330889 -0.4945464 -0.8173201
#> [3,]  1.787887 1.589313 -1.3248203  0.7710950
#> [4,]  1.690895 2.510157  1.2100308  1.4111385

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>           [,1]     [,2]       [,3]       [,4]
#> [1,] -0.422219 3.018452  0.3255044  1.9746192
#> [2,]  1.570685 1.330889 -0.4945464 -0.8173201
#> [3,]  1.787887 1.589313 -1.3248203  0.7710950
#> [4,]  1.690895 2.510157  1.2100308  1.4111385
```
