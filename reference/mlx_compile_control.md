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
#> [1,] -0.3267730 0.7013923 1.4025977 -0.0982492
#> [2,] -0.1475364 1.0505514 1.1627598  1.5120285
#> [3,]  1.8269181 2.1475120 0.5471148  2.7007790
#> [4,] -0.1330237 1.3361564 2.4275134  1.1844001

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>            [,1]      [,2]      [,3]       [,4]
#> [1,] -0.3267730 0.7013923 1.4025977 -0.0982492
#> [2,] -0.1475364 1.0505514 1.1627598  1.5120285
#> [3,]  1.8269181 2.1475120 0.5471148  2.7007790
#> [4,] -0.1330237 1.3361564 2.4275134  1.1844001
```
