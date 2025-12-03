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
#>             [,1]       [,2]      [,3]      [,4]
#> [1,] -0.01776576  0.9371266 1.6381496 0.9028367
#> [2,]  0.51358259  2.4550998 1.0165120 1.8115475
#> [3,]  1.57722306  1.3163478 0.4239877 1.0872693
#> [4,]  3.29711699 -0.2044780 1.3169361 0.1032146

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>             [,1]       [,2]      [,3]      [,4]
#> [1,] -0.01776576  0.9371266 1.6381496 0.9028367
#> [2,]  0.51358259  2.4550998 1.0165120 1.8115475
#> [3,]  1.57722306  1.3163478 0.4239877 1.0872693
#> [4,]  3.29711699 -0.2044780 1.3169361 0.1032146
```
