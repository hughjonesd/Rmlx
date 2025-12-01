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
#>           [,1]       [,2]      [,3]        [,4]
#> [1,]  1.547077 0.89788151 0.8575678  1.16423428
#> [2,]  1.405643 2.11363220 0.7332403 -0.08005857
#> [3,]  2.282166 0.06141067 0.3780386 -0.14912701
#> [4,] -1.005105 0.77809083 1.6211283 -2.18134713

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>           [,1]       [,2]      [,3]        [,4]
#> [1,]  1.547077 0.89788151 0.8575678  1.16423428
#> [2,]  1.405643 2.11363220 0.7332403 -0.08005857
#> [3,]  2.282166 0.06141067 0.3780386 -0.14912701
#> [4,] -1.005105 0.77809083 1.6211283 -2.18134713
```
