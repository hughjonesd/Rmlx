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
#>           [,1]        [,2]      [,3]       [,4]
#> [1,] 0.4813467  2.64345026 0.4233252 0.10939276
#> [2,] 0.3358225  2.86242580 0.7693932 1.11291444
#> [3,] 0.9723264 -0.09998143 1.3280067 0.08325809
#> [4,] 2.6428106  1.75009704 1.0821856 0.24238962

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   device: gpu
#>   values:
#>           [,1]        [,2]      [,3]       [,4]
#> [1,] 0.4813467  2.64345026 0.4233252 0.10939276
#> [2,] 0.3358225  2.86242580 0.7693932 1.11291444
#> [3,] 0.9723264 -0.09998143 1.3280067 0.08325809
#> [4,] 2.6428106  1.75009704 1.0821856 0.24238962
```
