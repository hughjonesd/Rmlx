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
#>             [,1]       [,2]        [,3]         [,4]
#> [1,]  0.44620705  1.4967663 -0.24902987  0.894947410
#> [2,]  0.04336250  1.0106324 -0.37806237 -0.134838820
#> [3,]  1.16520083  1.3694551  0.25915205  0.994017422
#> [4,] -0.07723117 -0.1378988  0.08737779  0.002861381

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>             [,1]       [,2]        [,3]         [,4]
#> [1,]  0.44620705  1.4967663 -0.24902987  0.894947410
#> [2,]  0.04336250  1.0106324 -0.37806237 -0.134838820
#> [3,]  1.16520083  1.3694551  0.25915205  0.994017422
#> [4,] -0.07723117 -0.1378988  0.08737779  0.002861381
```
