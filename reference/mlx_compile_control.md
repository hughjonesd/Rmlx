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
#>           [,1]       [,2]       [,3]        [,4]
#> [1,] 1.5266569  0.6711425  1.1843877 -0.56121016
#> [2,] 0.9097061 -0.6212531  0.5339196  0.54841173
#> [3,] 1.7114451  0.4042609  1.9843695 -0.19583380
#> [4,] 0.9960366  0.6537522 -1.1762862  0.09966999

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>           [,1]       [,2]       [,3]        [,4]
#> [1,] 1.5266569  0.6711425  1.1843877 -0.56121016
#> [2,] 0.9097061 -0.6212531  0.5339196  0.54841173
#> [3,] 1.7114451  0.4042609  1.9843695 -0.19583380
#> [4,] 0.9960366  0.6537522 -1.1762862  0.09966999
```
