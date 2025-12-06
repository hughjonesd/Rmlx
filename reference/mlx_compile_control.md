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
#>             [,1]       [,2]      [,3]       [,4]
#> [1,] -0.01446199 0.03761661 1.1766814  2.3520925
#> [2,]  0.39326245 2.09716320 0.4663464 -0.1151081
#> [3,]  2.43053341 0.57336140 1.5047040  2.2403002
#> [4,]  2.20264196 3.07708263 0.1785598  1.3118148

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>             [,1]       [,2]      [,3]       [,4]
#> [1,] -0.01446199 0.03761661 1.1766814  2.3520925
#> [2,]  0.39326245 2.09716320 0.4663464 -0.1151081
#> [3,]  2.43053341 0.57336140 1.5047040  2.2403002
#> [4,]  2.20264196 3.07708263 0.1785598  1.3118148
```
