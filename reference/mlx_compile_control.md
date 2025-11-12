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
#>             [,1]       [,2]       [,3]     [,4]
#> [1,]  0.09262139 -0.6912626  0.7408038 2.111843
#> [2,]  0.95556080  0.6327920  0.8961352 0.584092
#> [3,]  0.35882837  3.1549313 -0.5728279 1.134996
#> [4,] -0.13535726  0.7377070 -0.5601139 1.682477

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>             [,1]       [,2]       [,3]     [,4]
#> [1,]  0.09262139 -0.6912626  0.7408038 2.111843
#> [2,]  0.95556080  0.6327920  0.8961352 0.584092
#> [3,]  0.35882837  3.1549313 -0.5728279 1.134996
#> [4,] -0.13535726  0.7377070 -0.5601139 1.682477
```
