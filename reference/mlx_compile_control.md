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
#>            [,1]      [,2]       [,3]        [,4]
#> [1,]  2.1415784 1.6071209  0.5206838  2.80823851
#> [2,] -0.4662563 0.9839352 -0.4718921 -0.04029393
#> [3,]  0.4209805 1.2992256  0.2373906 -0.42259514
#> [4,] -0.4065006 0.4263426 -0.1711125  0.81127113

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>            [,1]      [,2]       [,3]        [,4]
#> [1,]  2.1415784 1.6071209  0.5206838  2.80823851
#> [2,] -0.4662563 0.9839352 -0.4718921 -0.04029393
#> [3,]  0.4209805 1.2992256  0.2373906 -0.42259514
#> [4,] -0.4065006 0.4263426 -0.1711125  0.81127113
```
