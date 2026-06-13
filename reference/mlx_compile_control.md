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
#>           [,1]       [,2]      [,3]       [,4]
#> [1,] 0.2581934 -0.7173959 0.5687370  1.4602776
#> [2,] 2.1101298 -0.1900183 2.5675745 -1.0178449
#> [3,] 1.7734675  0.3667749 0.3598099  3.0634420
#> [4,] 2.3973699  1.2058542 0.6233015  0.5864198

# Re-enable compilation
mlx_enable_compile()
demo_fn(x)  # Runs with optimization
#> mlx array [4 x 4]
#>   dtype: float32
#>   values:
#>           [,1]       [,2]      [,3]       [,4]
#> [1,] 0.2581934 -0.7173959 0.5687370  1.4602776
#> [2,] 2.1101298 -0.1900183 2.5675745 -1.0178449
#> [3,] 1.7734675  0.3667749 0.3598099  3.0634420
#> [4,] 2.3973699  1.2058542 0.6233015  0.5864198
```
