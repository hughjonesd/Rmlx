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
#>            [,1]      [,2]     [,3]      [,4]
#> [1,]  0.2780029 2.1443508 2.051080 0.4121697
#> [2,] -0.6903915 0.3730831 1.824584 0.2147855
#> [3,]  2.1246254 0.6456604 0.579968 0.6168704
#> [4,]  1.1711625 2.0714092 1.146502 0.8235092

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>            [,1]      [,2]     [,3]      [,4]
#> [1,]  0.2780029 2.1443508 2.051080 0.4121697
#> [2,] -0.6903915 0.3730831 1.824584 0.2147855
#> [3,]  2.1246254 0.6456604 0.579968 0.6168704
#> [4,]  1.1711625 2.0714092 1.146502 0.8235092
```
