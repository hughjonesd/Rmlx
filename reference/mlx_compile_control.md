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
#> [1,] 0.4960793 0.5583314 2.2107997  1.4426157
#> [2,] 0.9602202 1.1282755 1.0701700  2.9068336
#> [3,] 1.6377083 2.2695155 0.6202183 -0.3515787
#> [4,] 0.7348480 0.3983600 0.2793365  2.4060588

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>           [,1]      [,2]      [,3]       [,4]
#> [1,] 0.4960793 0.5583314 2.2107997  1.4426157
#> [2,] 0.9602202 1.1282755 1.0701700  2.9068336
#> [3,] 1.6377083 2.2695155 0.6202183 -0.3515787
#> [4,] 0.7348480 0.3983600 0.2793365  2.4060588
```
