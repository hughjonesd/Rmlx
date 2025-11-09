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
#>           [,1]      [,2]      [,3]       [,4]
#> [1,] 1.0688325 0.1315882 0.9023013 -1.0205338
#> [2,] 0.8892233 0.4993329 2.2142611  0.2672415
#> [3,] 0.6604408 1.8436093 1.1376525 -0.6052276
#> [4,] 1.9142716 0.9526029 2.4566865  1.6270421

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>           [,1]      [,2]      [,3]       [,4]
#> [1,] 1.0688325 0.1315882 0.9023013 -1.0205338
#> [2,] 0.8892233 0.4993329 2.2142611  0.2672415
#> [3,] 0.6604408 1.8436093 1.1376525 -0.6052276
#> [4,] 1.9142716 0.9526029 2.4566865  1.6270421
```
