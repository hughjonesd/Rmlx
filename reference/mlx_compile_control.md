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
#> [1,] 0.5116277 0.1715051 1.8281865 0.42240697
#> [2,] 0.7548473 0.2160609 0.3210369 3.05318761
#> [3,] 1.4516680 1.3705825 1.5497262 0.39025074
#> [4,] 3.0269599 1.4123161 1.5188431 0.07981783

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>           [,1]      [,2]      [,3]       [,4]
#> [1,] 0.5116277 0.1715051 1.8281865 0.42240697
#> [2,] 0.7548473 0.2160609 0.3210369 3.05318761
#> [3,] 1.4516680 1.3705825 1.5497262 0.39025074
#> [4,] 3.0269599 1.4123161 1.5188431 0.07981783
```
