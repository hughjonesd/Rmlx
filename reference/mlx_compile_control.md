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
#>            [,1]      [,2]      [,3]       [,4]
#> [1,] -0.1263561 2.2936788 1.0084578  2.8242054
#> [2,] -0.0893743 0.8561429 2.5289588  1.5357347
#> [3,]  2.5338802 0.1570947 0.7791643 -1.4242404
#> [4,]  2.2754519 1.0765177 1.8130827  0.3729913

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>            [,1]      [,2]      [,3]       [,4]
#> [1,] -0.1263561 2.2936788 1.0084578  2.8242054
#> [2,] -0.0893743 0.8561429 2.5289588  1.5357347
#> [3,]  2.5338802 0.1570947 0.7791643 -1.4242404
#> [4,]  2.2754519 1.0765177 1.8130827  0.3729913
```
