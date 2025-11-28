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
#>            [,1]        [,2]        [,3]        [,4]
#> [1,]  1.6790458  1.56584096  1.74871016  1.90019488
#> [2,]  2.6371827  0.06567198 -0.57232547  0.09253818
#> [3,]  1.6947426 -0.98524690  2.79972744 -0.60471475
#> [4,] -0.1965019  0.23739064  0.02640367  1.05171084

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>            [,1]        [,2]        [,3]        [,4]
#> [1,]  1.6790458  1.56584096  1.74871016  1.90019488
#> [2,]  2.6371827  0.06567198 -0.57232547  0.09253818
#> [3,]  1.6947426 -0.98524690  2.79972744 -0.60471475
#> [4,] -0.1965019  0.23739064  0.02640367  1.05171084
```
