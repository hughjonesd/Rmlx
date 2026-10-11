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
#>   values:
#>           [,1]       [,2]      [,3]      [,4]
#> [1,] 0.8050554  0.9157401 0.3552582 0.4024649
#> [2,] 1.6842034 -0.3087909 1.8265941 2.6887097
#> [3,] 0.9812306 -0.2716801 2.0164628 2.3247135
#> [4,] 1.4287895 -0.7919540 1.2353766 2.0157433

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   values:
#>           [,1]       [,2]      [,3]      [,4]
#> [1,] 0.8050554  0.9157401 0.3552582 0.4024649
#> [2,] 1.6842034 -0.3087909 1.8265941 2.6887097
#> [3,] 0.9812306 -0.2716801 2.0164628 2.3247135
#> [4,] 1.4287895 -0.7919540 1.2353766 2.0157433
```
